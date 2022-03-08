#include "TargetInfo.h"
#include "ABIInfo.h"
#include "CIRGenCXXABI.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenTypes.h"
#include "CallingConv.h"

#include "clang/Basic/TargetInfo.h"

using namespace cir;
using namespace clang;

namespace {

/// The AVX ABI leel for X86 targets.
enum class X86AVXABILevel { None, AVX, AVX512 };

class X86_64ABIInfo : public ABIInfo {
  enum Class {
    Integer = 0,
    SSE,
    SSEUp,
    X87,
    X87Up,
    ComplexX87,
    NoClass,
    Memory
  };

  // X86AVXABILevel AVXLevel;
  // Some ABIs (e.g. X32 ABI and Native Client OS) use 32 bit pointers on 64-bit
  // hardware.
  // bool Has64BitPointers;

public:
  X86_64ABIInfo(CIRGenTypes &CGT, X86AVXABILevel AVXLevel)
      : ABIInfo(CGT)
  // , AVXLevel(AVXLevel)
  // , Has64BitPointers(CGT.getDataLayout().getPointeSize(0) == 8)
  {}

  virtual void computeInfo(CIRGenFunctionInfo &FI) const override;

  /// classify - Determine the x86_64 register classes in which the given type T
  /// should be passed.
  ///
  /// \param Lo - The classification for the parts of the type residing in the
  /// low word of the containing object.
  ///
  /// \param Hi - The classification for the parts of the type residing in the
  /// high word of the containing object.
  ///
  /// \param OffsetBase - The bit offset of this type in the containing object.
  /// Some parameters are classified different depending on whether they
  /// straddle an eightbyte boundary.
  ///
  /// \param isNamedArg - Whether the argument in question is a "named"
  /// argument, as used in AMD64-ABI 3.5.7.
  ///
  /// If a word is unused its result will be NoClass; if a type should be passed
  /// in Memory then at least the classification of \arg Lo will be Memory.
  ///
  /// The \arg Lo class will be NoClass iff the argument is ignored.
  ///
  /// If the \arg Lo class is ComplexX87, then the \arg Hi class will also be
  /// ComplexX87.
  void classify(clang::QualType T, uint64_t OffsetBase, Class &Lo, Class &Hi,
                bool isNamedArg) const;

  ABIArgInfo classifyReturnType(QualType RetTy) const;

  ABIArgInfo classifyArgumentType(clang::QualType Ty, unsigned freeIntRegs,
                                  unsigned &neededInt, unsigned &neededSSE,
                                  bool isNamedArg) const;

  mlir::Type GetINTEGERTypeAtOffset(mlir::Type CIRType, unsigned CIROffset,
                                    QualType SourceTy,
                                    unsigned SourceOffset) const;

  /// getIndirectResult - Give a source type \arg Ty, return a suitable result
  /// such that the argument will be passed in memory.
  ///
  /// \param freeIntRegs - The number of free integer registers remaining
  /// available.
  ABIArgInfo getIndirectResult(QualType Ty, unsigned freeIntRegs) const;
};

class X86_64TargetCIRGenInfo : public TargetCIRGenInfo {
public:
  X86_64TargetCIRGenInfo(CIRGenTypes &CGT, X86AVXABILevel AVXLevel)
      : TargetCIRGenInfo(std::make_unique<X86_64ABIInfo>(CGT, AVXLevel)) {}
};
} // namespace

static bool classifyReturnType(const CIRGenCXXABI &CXXABI,
                               CIRGenFunctionInfo &FI, const ABIInfo &Info) {
  QualType Ty = FI.getReturnType();

  assert(!Ty->getAs<RecordType>() && "RecordType returns NYI");

  return CXXABI.classifyReturnType(FI);
}

CIRGenCXXABI &ABIInfo::getCXXABI() const { return CGT.getCXXABI(); }

ABIArgInfo X86_64ABIInfo::getIndirectResult(QualType Ty,
                                            unsigned freeIntRegs) const {
  assert(false && "NYI");
}

void X86_64ABIInfo::computeInfo(CIRGenFunctionInfo &FI) const {
  const unsigned CallingConv = FI.getCallingConvention();

  assert(CallingConv == cir::CallingConv::C && "C is the only supported CC");

  unsigned FreeIntRegs = 6;
  unsigned FreeSSERegs = 8;
  unsigned NeededInt, NeededSSE;

  assert(!::classifyReturnType(getCXXABI(), FI, *this) && "NYI");
  FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

  assert(!FI.getReturnInfo().isIndirect() && "Indirect return NYI");

  assert(!FI.isChainCall() && "Chain call NYI");

  unsigned NumRequiredArgs = FI.getNumRequiredArgs();
  // AMD64-ABI 3.2.3p3: Once arguments are classified, the registers get
  // assigned (in left-to-right order) for passing as follows...
  unsigned ArgNo = 0;
  for (CIRGenFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it, ++ArgNo) {
    bool IsNamedArg = ArgNo < NumRequiredArgs;

    assert(!it->type->isStructureOrClassType() && "NYI");

    it->info = classifyArgumentType(it->type, FreeIntRegs, NeededInt, NeededSSE,
                                    IsNamedArg);

    // AMD64-ABI 3.2.3p3: If there are no registers available for any eightbyte
    // of an argument, the whole argument is passed on the stack. If registers
    // have already been assigned for some eightbytes of such an argument, the
    // assignments get reverted.
    if (FreeIntRegs >= NeededInt && FreeSSERegs >= NeededSSE) {
      FreeIntRegs -= NeededInt;
      FreeSSERegs -= NeededSSE;
    } else {
      it->info = getIndirectResult(it->type, FreeIntRegs);
    }
  }
}

/// Pass transparent unions as if they were the type of the first element. Sema
/// should ensure that all elements of the union have the same "machine type".
static QualType useFirstFieldIfTransparentUnion(QualType Ty) {
  assert(!Ty->getAsUnionType() && "NYI");
  return Ty;
}

/// GetINTEGERTypeAtOffset - The ABI specifies that a value should be passed in
/// an 8-byte GPR. This means that we either have a scalar or we are talking
/// about the high or low part of an up-to-16-byte struct. This routine picks
/// the best CIR type to represent this, which may be i64 or may be anything
/// else that the backend will pass in a GPR that works better (e.g. i8, %foo*,
/// etc).
///
/// PrefType is a CIR type that corresponds to (part of) the IR type for the
/// source type. CIROffset is an offset in bytes into the CIR type taht the
/// 8-byte value references. PrefType may be null.
///
/// SourceTy is the source-level type for the entire argument. SourceOffset is
/// an offset into this that we're processing (which is always either 0 or 8).
///
mlir::Type X86_64ABIInfo::GetINTEGERTypeAtOffset(mlir::Type CIRType,
                                                 unsigned CIROffset,
                                                 QualType SourceTy,
                                                 unsigned SourceOffset) const {
  assert(CIROffset == 0 && "NYI");
  assert(SourceOffset == 0 && "NYI");
  // TODO: this entire function. It's safe to now just to let the integer type
  // be used as is since we aren't actually generating anything.
  return CIRType;
}

ABIArgInfo X86_64ABIInfo::classifyArgumentType(QualType Ty,
                                               unsigned int freeIntRegs,
                                               unsigned int &neededInt,
                                               unsigned int &neededSSE,
                                               bool isNamedArg) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  X86_64ABIInfo::Class Lo, Hi;
  classify(Ty, 0, Lo, Hi, isNamedArg);

  // Check some invariants
  // FIXME: Enforce these by construction.
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  neededInt = 0;
  neededSSE = 0;
  mlir::Type ResType = nullptr;
  switch (Lo) {
  default:
    assert(false && "NYI");

  // AMD64-ABI 3.2.3p3: Rule 2. If the class is INTEGER, the next available
  // register of the sequence %rdi, %rsi, %rdx, %rcx, %r8 and %r9 is used.
  case Integer:
    ++neededInt;

    // Pick an 8-byte type based on the preferred type.
    ResType = GetINTEGERTypeAtOffset(CGT.ConvertType(Ty), 0, Ty, 0);

    // If we have a sign or zero extended integer, make sure to return Extend so
    // that the parameter gets the right LLVM IR attributes.
    if (Hi == NoClass && ResType.isa<mlir::IntegerType>()) {
      assert(!Ty->getAs<EnumType>() && "NYI");
      assert(!isPromotableIntegerTypeForABI(Ty) && "NYI");
    }

    break;
  }

  mlir::Type HighPart = nullptr;
  switch (Hi) {
  default:
    assert(false && "NYI");
  case NoClass:
    break;
  }

  assert(!HighPart && "NYI");

  return ABIArgInfo::getDirect(ResType);
}

ABIInfo::~ABIInfo() {}

bool ABIInfo::isPromotableIntegerTypeForABI(QualType Ty) const {
  assert(false && "NYI");

  assert(!Ty->getAs<BitIntType>() && "NYI");

  return false;
}

void X86_64ABIInfo::classify(QualType Ty, uint64_t OffsetBase, Class &Lo,
                             Class &Hi, bool isNamedArg) const {
  Lo = Hi = NoClass;
  Class &Current = OffsetBase < 64 ? Lo : Hi;
  Current = Memory;

  auto *BT = Ty->getAs<BuiltinType>();
  assert(BT && "Only builtin types implemented.");
  BuiltinType::Kind k = BT->getKind();
  if (k == BuiltinType::Void)
    Current = NoClass;
  else if (k >= BuiltinType::Bool && k <= BuiltinType::LongLong) {
    Current = Integer;
  } else {
    assert(false && "Only void and Integer supported so far");
  }
  return;
}

ABIArgInfo X86_64ABIInfo::classifyReturnType(QualType RetTy) const {
  // AMD64-ABI 3.2.3p4: Rule 1. Classify the return type with the classification
  // algorithm.
  X86_64ABIInfo::Class Lo, Hi;
  classify(RetTy, 0, Lo, Hi, /*isNamedArg*/ true);

  // Check some invariants.
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  // mlir::Type ResType = nullptr;
  assert(Lo == NoClass && "Only NoClass Supported so far");
  assert(Hi == NoClass && "Only NoClass Supported so far");

  return ABIArgInfo::getIgnore();
}

const TargetCIRGenInfo &CIRGenModule::getTargetCIRGenInfo() {
  if (TheTargetCIRGenInfo)
    return *TheTargetCIRGenInfo;

  // Helper to set the unique_ptr while still keeping the return value.
  auto SetCIRGenInfo = [&](TargetCIRGenInfo *P) -> const TargetCIRGenInfo & {
    this->TheTargetCIRGenInfo.reset(P);
    return *P;
  };

  const llvm::Triple &Triple = getTarget().getTriple();

  switch (Triple.getArch()) {
  default:
    assert(false && "Target not yet supported!");
  case llvm::Triple::x86_64: {
    StringRef ABI = getTarget().getABI();
    X86AVXABILevel AVXLevel = (ABI == "avx512" ? X86AVXABILevel::AVX512
                               : ABI == "avx"  ? X86AVXABILevel::AVX
                                               : X86AVXABILevel::None);

    switch (Triple.getOS()) {
    default:
      assert(false && "OSType NYI");
    case llvm::Triple::Linux:
      return SetCIRGenInfo(new X86_64TargetCIRGenInfo(genTypes, AVXLevel));
    }
  }
  }
}
