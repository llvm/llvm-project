//===- DeviceThreadprivateCheck.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnose accesses to threadprivate variables in target device code and
// remove the threadprivate operations of unaccessed variables from it.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenMP/Passes.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace flangomp {
#define GEN_PASS_DEF_DEVICETHREADPRIVATECHECKPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

/// Check whether the given operation is part of target device code, i.e. it is
/// inside of an omp.target region or of a function that is declare target for
/// the device.
static bool isDeviceCode(Operation *op) {
  if (op->getParentOfType<omp::TargetOp>())
    return true;

  if (auto declareTargetOp = op->getParentOfType<omp::DeclareTargetInterface>())
    if (omp::DeclareTargetAttr declareTargetAttr =
            declareTargetOp.getDeclareTarget())
      return declareTargetAttr.getDeviceType() !=
             omp::DeclareTargetDeviceType::host;

  return false;
}

/// Operations that only compute an address or a descriptor from their operands
/// without accessing the memory behind them.
static bool isAddressComputation(Operation *op) {
  return isa<fir::FortranVariableOpInterface, fir::CoordinateOp, fir::ConvertOp,
             fir::EmboxOp, fir::ReboxOp, fir::BoxAddrOp, fir::ArrayCoorOp,
             hlfir::DesignateOp>(op);
}

/// Follow the uses of \p value through address computations, which are added
/// to \p addressOps. Return the first operation accessing the memory, if any.
static Operation *findAccess(Value value,
                             llvm::SetVector<Operation *> &addressOps) {
  Operation *access = nullptr;
  for (Operation *user : value.getUsers()) {
    if (!isAddressComputation(user)) {
      // Keep the access that comes first in the block, if possible.
      if (!access || (user->getBlock() == access->getBlock() &&
                      user->isBeforeInBlock(access)))
        access = user;
      continue;
    }
    if (!addressOps.insert(user))
      continue;
    for (Value result : user->getResults())
      if (Operation *resultAccess = findAccess(result, addressOps))
        if (!access || (resultAccess->getBlock() == access->getBlock() &&
                        resultAccess->isBeforeInBlock(access)))
          access = resultAccess;
  }
  return access;
}

/// Return the global holding the threadprivate variable of \p op, if found.
static fir::GlobalOp getThreadprivateGlobal(omp::ThreadprivateOp op) {
  Value addr = op.getSymAddr();
  while (Operation *defOp = addr.getDefiningOp()) {
    if (auto addrOf = dyn_cast<fir::AddrOfOp>(defOp))
      return SymbolTable::lookupNearestSymbolFrom<fir::GlobalOp>(
          op, addrOf.getSymbol());
    if (auto varOp = dyn_cast<fir::FortranVariableOpInterface>(defOp))
      addr = varOp.getBase();
    else if (auto convert = dyn_cast<fir::ConvertOp>(defOp))
      addr = convert.getValue();
    else
      break;
  }
  return nullptr;
}

/// Describe the threadprivate variable stored in \p global for a diagnostic.
static std::string describeThreadprivate(fir::GlobalOp global) {
  if (!global)
    return "a THREADPRIVATE variable";

  llvm::StringRef symName = global.getSymName();
  if (global.getLinkage() == fir::LinkageEnum::Common) {
    // Common block names are mangled like external procedure names.
    llvm::StringRef name = symName;
    if (name == "__BLNK__")
      name = "";
    else
      name.consume_back("_");
    return ("THREADPRIVATE common block '/" + name + "/'").str();
  }

  auto [kind, deconstructed] = fir::NameUniquer::deconstruct(symName);
  llvm::StringRef name = kind == fir::NameUniquer::NameKind::VARIABLE
                             ? deconstructed.name
                             : symName;
  return ("THREADPRIVATE variable '" + name + "'").str();
}

namespace {
class DeviceThreadprivateCheckPass
    : public flangomp::impl::DeviceThreadprivateCheckPassBase<
          DeviceThreadprivateCheckPass> {

  void runOnOperation() override {
    // Only run checks when compiling for a target device.
    auto offloadModule = dyn_cast<omp::OffloadModuleInterface>(*getOperation());
    if (!offloadModule || !offloadModule.getIsTargetDevice())
      return;

    llvm::SmallVector<omp::ThreadprivateOp> threadprivateOps;
    getOperation()->walk([&](omp::ThreadprivateOp op) {
      if (isDeviceCode(op))
        threadprivateOps.push_back(op);
    });

    bool errorsEmitted = false;
    for (omp::ThreadprivateOp op : threadprivateOps) {
      llvm::SetVector<Operation *> addressOps;
      if (Operation *access = findAccess(op.getTlsAddr(), addressOps)) {
        fir::GlobalOp global = getThreadprivateGlobal(op);
        InFlightDiagnostic diag = access->emitError()
                                  << describeThreadprivate(global)
                                  << " cannot be accessed in target device "
                                     "code";
        if (global)
          diag.attachNote(global.getLoc()) << "declared here";
        errorsEmitted = true;
        continue;
      }

      // The variable is declared, but never accessed. Remove the threadprivate
      // operation, which cannot be translated for the device, together with
      // the address computations depending on it. All users of these address
      // computations are in the set, so there is always one without remaining
      // uses that can be erased.
      llvm::SmallVector<Operation *> pending(addressOps.begin(),
                                             addressOps.end());
      while (!pending.empty()) {
        auto *unused = llvm::find_if(pending, [](Operation *addressOp) {
          return addressOp->use_empty();
        });
        assert(unused != pending.end() && "expected an unused operation");
        (*unused)->erase();
        pending.erase(unused);
      }
      op->erase();
    }

    if (errorsEmitted)
      signalPassFailure();
  }
};
} // namespace
