
#include "clang/CIR/Target/x86.h"
#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "LowerModule.h"
#include "LowerTypes.h"
#include "TargetInfo.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

namespace mlir {
namespace cir {

class X86_64ABIInfo : public ABIInfo {
  using Class = ::cir::X86ArgClass;

  /// Determine the x86_64 register classes in which the given type T should be
  /// passed.
  ///
  /// \param Lo - The classification for the parts of the type
  /// residing in the low word of the containing object.
  ///
  /// \param Hi - The classification for the parts of the type
  /// residing in the high word of the containing object.
  ///
  /// \param OffsetBase - The bit offset of this type in the
  /// containing object.  Some parameters are classified different
  /// depending on whether they straddle an eightbyte boundary.
  ///
  /// \param isNamedArg - Whether the argument in question is a "named"
  /// argument, as used in AMD64-ABI 3.5.7.
  ///
  /// \param IsRegCall - Whether the calling conversion is regcall.
  ///
  /// If a word is unused its result will be NoClass; if a type should
  /// be passed in Memory then at least the classification of \arg Lo
  /// will be Memory.
  ///
  /// The \arg Lo class will be NoClass iff the argument is ignored.
  ///
  /// If the \arg Lo class is ComplexX87, then the \arg Hi class will
  /// also be ComplexX87.
  void classify(Type T, uint64_t OffsetBase, Class &Lo, Class &Hi,
                bool isNamedArg, bool IsRegCall = false) const;

public:
  X86_64ABIInfo(LowerTypes &CGT, X86AVXABILevel AVXLevel) : ABIInfo(CGT) {}

  ::cir::ABIArgInfo classifyReturnType(Type RetTy) const;

  void computeInfo(LowerFunctionInfo &FI) const override;
};

class X86_64TargetLoweringInfo : public TargetLoweringInfo {
public:
  X86_64TargetLoweringInfo(LowerTypes &LM, X86AVXABILevel AVXLevel)
      : TargetLoweringInfo(std::make_unique<X86_64ABIInfo>(LM, AVXLevel)) {
    assert(!::cir::MissingFeatures::swift());
  }
};

void X86_64ABIInfo::classify(Type Ty, uint64_t OffsetBase, Class &Lo, Class &Hi,
                             bool isNamedArg, bool IsRegCall) const {
  // FIXME: This code can be simplified by introducing a simple value class
  // for Class pairs with appropriate constructor methods for the various
  // situations.

  // FIXME: Some of the split computations are wrong; unaligned vectors
  // shouldn't be passed in registers for example, so there is no chance they
  // can straddle an eightbyte. Verify & simplify.

  Lo = Hi = Class::NoClass;

  Class &Current = OffsetBase < 64 ? Lo : Hi;
  Current = Class::Memory;

  // FIXME(cir): There's currently no direct way to identify if a type is a
  // builtin.
  if (/*isBuitinType=*/true) {
    if (Ty.isa<VoidType>()) {
      Current = Class::NoClass;
    } else {
      llvm::outs() << "Missing X86 classification for type " << Ty << "\n";
      llvm_unreachable("NYI");
    }
    // FIXME: _Decimal32 and _Decimal64 are SSE.
    // FIXME: _float128 and _Decimal128 are (SSE, SSEUp).
    return;
  }

  llvm::outs() << "Missing X86 classification for non-builtin types\n";
  llvm_unreachable("NYI");
}

::cir::ABIArgInfo X86_64ABIInfo::classifyReturnType(Type RetTy) const {
  // AMD64-ABI 3.2.3p4: Rule 1. Classify the return type with the
  // classification algorithm.
  X86_64ABIInfo::Class Lo, Hi;
  classify(RetTy, 0, Lo, Hi, true);

  // Check some invariants.
  assert((Hi != Class::Memory || Lo == Class::Memory) &&
         "Invalid memory classification.");
  assert((Hi != Class::SSEUp || Lo == Class::SSE) &&
         "Invalid SSEUp classification.");

  switch (Lo) {
  case Class::NoClass:
    if (Hi == Class::NoClass)
      return ::cir::ABIArgInfo::getIgnore();
    break;
  default:
    llvm_unreachable("NYI");
  }

  llvm_unreachable("NYI");
}

void X86_64ABIInfo::computeInfo(LowerFunctionInfo &FI) const {
  const unsigned CallingConv = FI.getCallingConvention();
  // It is possible to force Win64 calling convention on any x86_64 target by
  // using __attribute__((ms_abi)). In such case to correctly emit Win64
  // compatible code delegate this call to WinX86_64ABIInfo::computeInfo.
  if (CallingConv == llvm::CallingConv::Win64) {
    llvm_unreachable("Win64 CC is NYI");
  }

  bool IsRegCall = CallingConv == llvm::CallingConv::X86_RegCall;

  // Keep track of the number of assigned registers.
  unsigned NeededSSE = 0, MaxVectorWidth = 0;

  if (!::mlir::cir::classifyReturnType(getCXXABI(), FI, *this)) {
    if (IsRegCall || ::cir::MissingFeatures::regCall()) {
      llvm_unreachable("RegCall is NYI");
    } else
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
  }

  // If the return value is indirect, then the hidden argument is consuming
  // one integer register.
  if (FI.getReturnInfo().isIndirect())
    llvm_unreachable("NYI");
  else if (NeededSSE && MaxVectorWidth)
    llvm_unreachable("NYI");

  // The chain argument effectively gives us another free register.
  if (::cir::MissingFeatures::chainCall())
    llvm_unreachable("NYI");

  // AMD64-ABI 3.2.3p3: Once arguments are classified, the registers
  // get assigned (in left-to-right order) for passing as follows...
  unsigned ArgNo = 0;
  for (LowerFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it, ++ArgNo) {
    llvm_unreachable("NYI");
  }
}

std::unique_ptr<TargetLoweringInfo>
createX86_64TargetLoweringInfo(LowerModule &LM, X86AVXABILevel AVXLevel) {
  return std::make_unique<X86_64TargetLoweringInfo>(LM.getTypes(), AVXLevel);
}

} // namespace cir
} // namespace mlir
