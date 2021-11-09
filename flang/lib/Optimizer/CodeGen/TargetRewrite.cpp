//===-- TargetRewrite.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Target rewrite: rewriting of ops to make target-specific lowerings manifest.
// LLVM expects different lowering idioms to be used for distinct target
// triples. These distinctions are handled by this pass.
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "Target.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace fir;

#define DEBUG_TYPE "flang-target-rewrite"

namespace {

/// Fixups for updating a FuncOp's arguments and return values.
struct FixupTy {
  enum class Codes { CharPair, Trailing };

  FixupTy(Codes code, std::size_t index, std::size_t second = 0)
      : code{code}, index{index}, second{second} {}
  FixupTy(Codes code, std::size_t index,
          std::function<void(mlir::FuncOp)> &&finalizer)
      : code{code}, index{index}, finalizer{finalizer} {}
  FixupTy(Codes code, std::size_t index, std::size_t second,
          std::function<void(mlir::FuncOp)> &&finalizer)
      : code{code}, index{index}, second{second}, finalizer{finalizer} {}

  Codes code;
  std::size_t index;
  std::size_t second{};
  llvm::Optional<std::function<void(mlir::FuncOp)>> finalizer{};
}; // namespace

/// Target-specific rewriting of the FIR. This is a prerequisite pass to code
/// generation that traverses the FIR and modifies types and operations to a
/// form that is appropriate for the specific target. LLVM IR has specific
/// idioms that are used for distinct target processor and ABI combinations.
class TargetRewrite : public TargetRewriteBase<TargetRewrite> {
public:
  TargetRewrite(const TargetRewriteOptions &options) {
    noCharacterConversion = options.noCharacterConversion;
  }

  void runOnOperation() override final {
    auto &context = getContext();
    mlir::OpBuilder rewriter(&context);

    auto mod = getModule();
    if (!forcedTargetTriple.empty()) {
      setTargetTriple(mod, forcedTargetTriple);
    }

    auto specifics = CodeGenSpecifics::get(getOperation().getContext(),
                                           getTargetTriple(getOperation()),
                                           getKindMapping(getOperation()));
    setMembers(specifics.get(), &rewriter);

    // Perform type conversion on signatures and call sites.
    if (mlir::failed(convertTypes(mod))) {
      mlir::emitError(mlir::UnknownLoc::get(&context),
                      "error in converting types to target abi");
      signalPassFailure();
    }

    // Convert ops in target-specific patterns.
    mod.walk([&](mlir::Operation *op) {
      if (auto call = dyn_cast<fir::CallOp>(op)) {
        if (!hasPortableSignature(call.getFunctionType()))
          convertCallOp(call);
      } else if (auto dispatch = dyn_cast<DispatchOp>(op)) {
        if (!hasPortableSignature(dispatch.getFunctionType()))
          convertCallOp(dispatch);
      }
    });

    clearMembers();
  }

  mlir::ModuleOp getModule() { return getOperation(); }

  // Convert fir.call and fir.dispatch Ops.
  template <typename A>
  void convertCallOp(A callOp) {
    auto fnTy = callOp.getFunctionType();
    auto loc = callOp.getLoc();
    rewriter->setInsertionPoint(callOp);
    llvm::SmallVector<mlir::Type> newResTys;
    llvm::SmallVector<mlir::Type> newInTys;
    llvm::SmallVector<mlir::Value> newOpers;

    // If the call is indirect, the first argument must still be the function
    // to call.
    int dropFront = 0;
    if constexpr (std::is_same_v<std::decay_t<A>, fir::CallOp>) {
      if (!callOp.callee().hasValue()) {
        newInTys.push_back(fnTy.getInput(0));
        newOpers.push_back(callOp.getOperand(0));
        dropFront = 1;
      }
    }

    // Determine the rewrite function, `wrap`, for the result value.
    llvm::Optional<std::function<mlir::Value(mlir::Operation *)>> wrap;
    if (fnTy.getResults().size() == 1) {
      mlir::Type ty = fnTy.getResult(0);
      newResTys.push_back(ty);
    } else if (fnTy.getResults().size() > 1) {
      TODO(loc, "multiple results not supported yet");
    }

    llvm::SmallVector<mlir::Type> trailingInTys;
    llvm::SmallVector<mlir::Value> trailingOpers;
    for (auto e : llvm::enumerate(
             llvm::zip(fnTy.getInputs().drop_front(dropFront),
                       callOp.getOperands().drop_front(dropFront)))) {
      mlir::Type ty = std::get<0>(e.value());
      mlir::Value oper = std::get<1>(e.value());
      unsigned index = e.index();
      llvm::TypeSwitch<mlir::Type>(ty)
          .template Case<BoxCharType>([&](BoxCharType boxTy) {
            bool sret;
            if constexpr (std::is_same_v<std::decay_t<A>, fir::CallOp>) {
              sret = callOp.callee() &&
                     functionArgIsSRet(index,
                                       getModule().lookupSymbol<mlir::FuncOp>(
                                           *callOp.callee()));
            } else {
              // TODO: dispatch case; how do we put arguments on a call?
              // We cannot put both an sret and the dispatch object first.
              sret = false;
              TODO(loc, "dispatch + sret not supported yet");
            }
            auto m = specifics->boxcharArgumentType(boxTy.getEleTy(), sret);
            auto unbox =
                rewriter->create<UnboxCharOp>(loc, std::get<mlir::Type>(m[0]),
                                              std::get<mlir::Type>(m[1]), oper);
            // unboxed CHARACTER arguments
            for (auto e : llvm::enumerate(m)) {
              unsigned idx = e.index();
              auto attr = std::get<CodeGenSpecifics::Attributes>(e.value());
              auto argTy = std::get<mlir::Type>(e.value());
              if (attr.isAppend()) {
                trailingInTys.push_back(argTy);
                trailingOpers.push_back(unbox.getResult(idx));
              } else {
                newInTys.push_back(argTy);
                newOpers.push_back(unbox.getResult(idx));
              }
            }
          })
          .Default([&](mlir::Type ty) {
            newInTys.push_back(ty);
            newOpers.push_back(oper);
          });
    }
    newInTys.insert(newInTys.end(), trailingInTys.begin(), trailingInTys.end());
    newOpers.insert(newOpers.end(), trailingOpers.begin(), trailingOpers.end());
    if constexpr (std::is_same_v<std::decay_t<A>, fir::CallOp>) {
      fir::CallOp newCall;
      if (callOp.callee().hasValue()) {
        newCall = rewriter->create<A>(loc, callOp.callee().getValue(),
                                      newResTys, newOpers);
      } else {
        // Force new type on the input operand.
        newOpers[0].setType(mlir::FunctionType::get(
            callOp.getContext(),
            mlir::TypeRange{newInTys}.drop_front(dropFront), newResTys));
        newCall = rewriter->create<A>(loc, newResTys, newOpers);
      }
      LLVM_DEBUG(llvm::dbgs() << "replacing call with " << newCall << '\n');
      if (wrap.hasValue())
        replaceOp(callOp, (*wrap)(newCall.getOperation()));
      else
        replaceOp(callOp, newCall.getResults());
    } else {
      // A is fir::DispatchOp
      TODO(loc, "dispatch not implemented");
    }
  }
  /// Convert the type signatures on all the functions present in the module.
  /// As the type signature is being changed, this must also update the
  /// function itself to use any new arguments, etc.
  mlir::LogicalResult convertTypes(mlir::ModuleOp mod) {
    for (auto fn : mod.getOps<mlir::FuncOp>())
      convertSignature(fn);
    return mlir::success();
  }

  /// If the signature does not need any special target-specific converions,
  /// then it is considered portable for any target, and this function will
  /// return `true`. Otherwise, the signature is not portable and `false` is
  /// returned.
  bool hasPortableSignature(mlir::Type signature) {
    assert(signature.isa<mlir::FunctionType>());
    auto func = signature.dyn_cast<mlir::FunctionType>();
    for (auto ty : func.getResults())
      if ((ty.isa<BoxCharType>() && !noCharacterConversion)) {
        LLVM_DEBUG(llvm::dbgs() << "rewrite " << signature << " for target\n");
        return false;
      }
    for (auto ty : func.getInputs())
      if ((ty.isa<BoxCharType>() && !noCharacterConversion)) {
        LLVM_DEBUG(llvm::dbgs() << "rewrite " << signature << " for target\n");
        return false;
      }
    return true;
  }

  /// Rewrite the signatures and body of the `FuncOp`s in the module for
  /// the immediately subsequent target code gen.
  void convertSignature(mlir::FuncOp func) {
    auto funcTy = func.getType().cast<mlir::FunctionType>();
    if (hasPortableSignature(funcTy))
      return;
    llvm::SmallVector<mlir::Type> newResTys;
    llvm::SmallVector<mlir::Type> newInTys;
    llvm::SmallVector<FixupTy> fixups;

    // Convert return value(s)
    for (auto ty : funcTy.getResults())
      newResTys.push_back(ty);

    // Convert arguments
    llvm::SmallVector<mlir::Type> trailingTys;
    for (auto e : llvm::enumerate(funcTy.getInputs())) {
      auto ty = e.value();
      unsigned index = e.index();
      llvm::TypeSwitch<mlir::Type>(ty)
          .Case<BoxCharType>([&](BoxCharType boxTy) {
            if (noCharacterConversion) {
              newInTys.push_back(boxTy);
            } else {
              // Convert a CHARACTER argument type. This can involve separating
              // the pointer and the LEN into two arguments and moving the LEN
              // argument to the end of the arg list.
              bool sret = functionArgIsSRet(index, func);
              for (auto e : llvm::enumerate(specifics->boxcharArgumentType(
                       boxTy.getEleTy(), sret))) {
                auto &tup = e.value();
                auto index = e.index();
                auto attr = std::get<CodeGenSpecifics::Attributes>(tup);
                auto argTy = std::get<mlir::Type>(tup);
                if (attr.isAppend()) {
                  trailingTys.push_back(argTy);
                } else {
                  if (sret) {
                    fixups.emplace_back(FixupTy::Codes::CharPair,
                                        newInTys.size(), index);
                  } else {
                    fixups.emplace_back(FixupTy::Codes::Trailing,
                                        newInTys.size(), trailingTys.size());
                  }
                  newInTys.push_back(argTy);
                }
              }
            }
          })
          .Default([&](mlir::Type ty) { newInTys.push_back(ty); });
    }

    if (!func.empty()) {
      // If the function has a body, then apply the fixups to the arguments and
      // return ops as required. These fixups are done in place.
      auto loc = func.getLoc();
      const auto fixupSize = fixups.size();
      const auto oldArgTys = func.getType().getInputs();
      int offset = 0;
      for (std::remove_const_t<decltype(fixupSize)> i = 0; i < fixupSize; ++i) {
        const auto &fixup = fixups[i];
        switch (fixup.code) {
        case FixupTy::Codes::CharPair: {
          // The FIR boxchar argument has been split into a pair of distinct
          // arguments that are in juxtaposition to each other.
          auto newArg =
              func.front().insertArgument(fixup.index, newInTys[fixup.index]);
          if (fixup.second == 1) {
            rewriter->setInsertionPointToStart(&func.front());
            auto boxTy = oldArgTys[fixup.index - offset - fixup.second];
            auto box = rewriter->create<EmboxCharOp>(
                loc, boxTy, func.front().getArgument(fixup.index - 1), newArg);
            func.getArgument(fixup.index + 1).replaceAllUsesWith(box);
            func.front().eraseArgument(fixup.index + 1);
            offset++;
          }
        } break;
        case FixupTy::Codes::Trailing: {
          // The FIR argument has been split into a pair of distinct arguments.
          // The first part of the pair appears in the original argument
          // position. The second part of the pair is appended after all the
          // original arguments. (Boxchar arguments.)
          auto newBufArg =
              func.front().insertArgument(fixup.index, newInTys[fixup.index]);
          auto newLenArg = func.front().addArgument(trailingTys[fixup.second]);
          auto boxTy = oldArgTys[fixup.index - offset];
          rewriter->setInsertionPointToStart(&func.front());
          auto box =
              rewriter->create<EmboxCharOp>(loc, boxTy, newBufArg, newLenArg);
          func.getArgument(fixup.index + 1).replaceAllUsesWith(box);
          func.front().eraseArgument(fixup.index + 1);
        } break;
        }
      }
    }

    // Set the new type and finalize the arguments, etc.
    newInTys.insert(newInTys.end(), trailingTys.begin(), trailingTys.end());
    auto newFuncTy =
        mlir::FunctionType::get(func.getContext(), newInTys, newResTys);
    LLVM_DEBUG(llvm::dbgs() << "new func: " << newFuncTy << '\n');
    func.setType(newFuncTy);

    for (auto &fixup : fixups)
      if (fixup.finalizer)
        (*fixup.finalizer)(func);
  }

  inline bool functionArgIsSRet(unsigned index, mlir::FuncOp func) {
    if (auto attr = func.getArgAttrOfType<mlir::UnitAttr>(index, "llvm.sret"))
      return true;
    return false;
  }

private:
  // Replace `op` and remove it.
  void replaceOp(mlir::Operation *op, mlir::ValueRange newValues) {
    op->replaceAllUsesWith(newValues);
    op->dropAllReferences();
    op->erase();
  }

  inline void setMembers(CodeGenSpecifics *s, mlir::OpBuilder *r) {
    specifics = s;
    rewriter = r;
  }

  inline void clearMembers() { setMembers(nullptr, nullptr); }

  CodeGenSpecifics *specifics{};
  mlir::OpBuilder *rewriter;
}; // namespace
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
fir::createFirTargetRewritePass(const TargetRewriteOptions &options) {
  return std::make_unique<TargetRewrite>(options);
}
