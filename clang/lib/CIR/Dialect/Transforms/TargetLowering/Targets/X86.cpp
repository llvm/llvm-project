
#include "clang/CIR/Target/x86.h"
#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "LowerModule.h"
#include "LowerTypes.h"
#include "TargetInfo.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

using X86AVXABILevel = ::cir::X86AVXABILevel;
using ABIArgInfo = ::cir::ABIArgInfo;

namespace mlir {
namespace cir {

namespace {

/// \p returns the size in bits of the largest (native) vector for \p AVXLevel.
unsigned getNativeVectorSizeForAVXABI(X86AVXABILevel AVXLevel) {
  switch (AVXLevel) {
  case X86AVXABILevel::AVX512:
    return 512;
  case X86AVXABILevel::AVX:
    return 256;
  case X86AVXABILevel::None:
    return 128;
  }
  llvm_unreachable("Unknown AVXLevel");
}

/// Return true if the specified [start,end) bit range is known to either be
/// off the end of the specified type or being in alignment padding.  The user
/// type specified is known to be at most 128 bits in size, and have passed
/// through X86_64ABIInfo::classify with a successful classification that put
/// one of the two halves in the INTEGER class.
///
/// It is conservatively correct to return false.
static bool BitsContainNoUserData(Type Ty, unsigned StartBit, unsigned EndBit,
                                  CIRLowerContext &Context) {
  // If the bytes being queried are off the end of the type, there is no user
  // data hiding here.  This handles analysis of builtins, vectors and other
  // types that don't contain interesting padding.
  unsigned TySize = (unsigned)Context.getTypeSize(Ty);
  if (TySize <= StartBit)
    return true;

  if (auto arrTy = llvm::dyn_cast<ArrayType>(Ty)) {
    llvm_unreachable("NYI");
  }

  if (auto structTy = llvm::dyn_cast<StructType>(Ty)) {
    const CIRRecordLayout &Layout = Context.getCIRRecordLayout(Ty);

    // If this is a C++ record, check the bases first.
    if (::cir::MissingFeatures::isCXXRecordDecl() ||
        ::cir::MissingFeatures::getCXXRecordBases()) {
      llvm_unreachable("NYI");
    }

    // Verify that no field has data that overlaps the region of interest. Yes
    // this could be sped up a lot by being smarter about queried fields,
    // however we're only looking at structs up to 16 bytes, so we don't care
    // much.
    unsigned idx = 0;
    for (auto type : structTy.getMembers()) {
      unsigned FieldOffset = (unsigned)Layout.getFieldOffset(idx);

      // If we found a field after the region we care about, then we're done.
      if (FieldOffset >= EndBit)
        break;

      unsigned FieldStart = FieldOffset < StartBit ? StartBit - FieldOffset : 0;
      if (!BitsContainNoUserData(type, FieldStart, EndBit - FieldOffset,
                                 Context))
        return false;

      ++idx;
    }

    // If nothing in this record overlapped the area of interest, we're good.
    return true;
  }

  return false;
}

/// Return a floating point type at the specified offset.
Type getFPTypeAtOffset(Type IRType, unsigned IROffset,
                       const ::cir::CIRDataLayout &TD) {
  if (IROffset == 0 && isa<SingleType, DoubleType>(IRType))
    return IRType;

  llvm_unreachable("NYI");
}

} // namespace

class X86_64ABIInfo : public ABIInfo {
  using Class = ::cir::X86ArgClass;

  /// Implement the X86_64 ABI merging algorithm.
  ///
  /// Merge an accumulating classification \arg Accum with a field
  /// classification \arg Field.
  ///
  /// \param Accum - The accumulating classification. This should
  /// always be either NoClass or the result of a previous merge
  /// call. In addition, this should never be Memory (the caller
  /// should just return Memory for the aggregate).
  static Class merge(Class Accum, Class Field);

  /// Implement the X86_64 ABI post merging algorithm.
  ///
  /// Post merger cleanup, reduces a malformed Hi and Lo pair to
  /// final MEMORY or SSE classes when necessary.
  ///
  /// \param AggregateSize - The size of the current aggregate in
  /// the classification process.
  ///
  /// \param Lo - The classification for the parts of the type
  /// residing in the low word of the containing object.
  ///
  /// \param Hi - The classification for the parts of the type
  /// residing in the higher words of the containing object.
  ///
  void postMerge(unsigned AggregateSize, Class &Lo, Class &Hi) const;

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

  Type GetSSETypeAtOffset(Type IRType, unsigned IROffset, Type SourceTy,
                          unsigned SourceOffset) const;

  Type GetINTEGERTypeAtOffset(Type DestTy, unsigned IROffset, Type SourceTy,
                              unsigned SourceOffset) const;

  /// The 0.98 ABI revision clarified a lot of ambiguities,
  /// unfortunately in ways that were not always consistent with
  /// certain previous compilers.  In particular, platforms which
  /// required strict binary compatibility with older versions of GCC
  /// may need to exempt themselves.
  bool honorsRevision0_98() const {
    return !getTarget().getTriple().isOSDarwin();
  }

  X86AVXABILevel AVXLevel;

public:
  X86_64ABIInfo(LowerTypes &CGT, X86AVXABILevel AVXLevel)
      : ABIInfo(CGT), AVXLevel(AVXLevel) {}

  ::cir::ABIArgInfo classifyReturnType(Type RetTy) const;

  ABIArgInfo classifyArgumentType(Type Ty, unsigned freeIntRegs,
                                  unsigned &neededInt, unsigned &neededSSE,
                                  bool isNamedArg, bool IsRegCall) const;

  void computeInfo(LowerFunctionInfo &FI) const override;
};

class X86_64TargetLoweringInfo : public TargetLoweringInfo {
public:
  X86_64TargetLoweringInfo(LowerTypes &LM, X86AVXABILevel AVXLevel)
      : TargetLoweringInfo(std::make_unique<X86_64ABIInfo>(LM, AVXLevel)) {
    assert(!::cir::MissingFeatures::swift());
  }

  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      mlir::cir::AddressSpaceAttr addressSpaceAttr) const override {
    using Kind = mlir::cir::AddressSpaceAttr::Kind;
    switch (addressSpaceAttr.getValue()) {
    case Kind::offload_private:
    case Kind::offload_local:
    case Kind::offload_global:
    case Kind::offload_constant:
    case Kind::offload_generic:
      return 0;
    default:
      llvm_unreachable("Unknown CIR address space for this target");
    }
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
    if (isa<VoidType>(Ty)) {
      Current = Class::NoClass;
    } else if (isa<IntType>(Ty)) {

      // FIXME(cir): Clang's BuiltinType::Kind allow comparisons (GT, LT, etc).
      // We should implement this in CIR to simplify the conditions below. BTW,
      // I'm not sure if the comparisons below are truly equivalent to the ones
      // in Clang.
      if (isa<IntType>(Ty)) {
        Current = Class::Integer;
      }
      return;

    } else if (isa<SingleType>(Ty) || isa<DoubleType>(Ty)) {
      Current = Class::SSE;
      return;

    } else if (isa<BoolType>(Ty)) {
      Current = Class::Integer;
    } else if (const auto RT = dyn_cast<StructType>(Ty)) {
      uint64_t Size = getContext().getTypeSize(Ty);

      // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger
      // than eight eightbytes, ..., it has class MEMORY.
      if (Size > 512)
        llvm_unreachable("NYI");

      // AMD64-ABI 3.2.3p2: Rule 2. If a C++ object has either a non-trivial
      // copy constructor or a non-trivial destructor, it is passed by invisible
      // reference.
      if (getRecordArgABI(RT, getCXXABI()))
        llvm_unreachable("NYI");

      // Assume variable sized types are passed in memory.
      if (::cir::MissingFeatures::recordDeclHasFlexibleArrayMember())
        llvm_unreachable("NYI");

      const auto &Layout = getContext().getCIRRecordLayout(Ty);

      // Reset Lo class, this will be recomputed.
      Current = Class::NoClass;

      // If this is a C++ record, classify the bases first.
      assert(!::cir::MissingFeatures::isCXXRecordDecl() &&
             !::cir::MissingFeatures::getCXXRecordBases());

      // Classify the fields one at a time, merging the results.
      bool UseClang11Compat = getContext().getLangOpts().getClangABICompat() <=
                                  clang::LangOptions::ClangABI::Ver11 ||
                              getContext().getTargetInfo().getTriple().isPS();
      bool IsUnion = RT.isUnion() && !UseClang11Compat;

      // FIXME(cir): An interface to handle field declaration might be needed.
      assert(!::cir::MissingFeatures::fieldDeclAbstraction());
      for (auto [idx, FT] : llvm::enumerate(RT.getMembers())) {
        uint64_t Offset = OffsetBase + Layout.getFieldOffset(idx);
        assert(!::cir::MissingFeatures::fieldDeclIsBitfield());
        bool BitField = false;

        // Ignore padding bit-fields.
        if (BitField && !::cir::MissingFeatures::fieldDeclisUnnamedBitField())
          llvm_unreachable("NYI");

        // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger than
        // eight eightbytes, or it contains unaligned fields, it has class
        // MEMORY.
        //
        // The only case a 256-bit or a 512-bit wide vector could be used is
        // when the struct contains a single 256-bit or 512-bit element. Early
        // check and fallback to memory.
        //
        // FIXME: Extended the Lo and Hi logic properly to work for size wider
        // than 128.
        if (Size > 128 && ((!IsUnion && Size != getContext().getTypeSize(FT)) ||
                           Size > getNativeVectorSizeForAVXABI(AVXLevel))) {
          llvm_unreachable("NYI");
        }
        // Note, skip this test for bit-fields, see below.
        if (!BitField && Offset % getContext().getTypeAlign(RT)) {
          llvm_unreachable("NYI");
        }

        // Classify this field.
        //
        // AMD64-ABI 3.2.3p2: Rule 3. If the size of the aggregate
        // exceeds a single eightbyte, each is classified
        // separately. Each eightbyte gets initialized to class
        // NO_CLASS.
        Class FieldLo, FieldHi;

        // Bit-fields require special handling, they do not force the
        // structure to be passed in memory even if unaligned, and
        // therefore they can straddle an eightbyte.
        if (BitField) {
          llvm_unreachable("NYI");
        } else {
          classify(FT, Offset, FieldLo, FieldHi, isNamedArg);
        }
        Lo = merge(Lo, FieldLo);
        Hi = merge(Hi, FieldHi);
        if (Lo == Class::Memory || Hi == Class::Memory)
          break;
      }

      postMerge(Size, Lo, Hi);
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

/// Return a type that will be passed by the backend in the low 8 bytes of an
/// XMM register, corresponding to the SSE class.
Type X86_64ABIInfo::GetSSETypeAtOffset(Type IRType, unsigned IROffset,
                                       Type SourceTy,
                                       unsigned SourceOffset) const {
  const ::cir::CIRDataLayout &TD = getDataLayout();
  unsigned SourceSize =
      (unsigned)getContext().getTypeSize(SourceTy) / 8 - SourceOffset;
  Type T0 = getFPTypeAtOffset(IRType, IROffset, TD);
  if (!T0 || isa<Float64Type>(T0))
    return T0; // NOTE(cir): Not sure if this is correct.

  Type T1 = {};
  unsigned T0Size = TD.getTypeAllocSize(T0);
  if (SourceSize > T0Size)
    llvm_unreachable("NYI");
  if (T1 == nullptr) {
    // Check if IRType is a half/bfloat + float. float type will be in
    // IROffset+4 due to its alignment.
    if (isa<Float16Type>(T0) && SourceSize > 4)
      llvm_unreachable("NYI");
    // If we can't get a second FP type, return a simple half or float.
    // avx512fp16-abi.c:pr51813_2 shows it works to return float for
    // {float, i8} too.
    if (T1 == nullptr)
      return T0;
  }

  llvm_unreachable("NYI");
}

/// The ABI specifies that a value should be passed in an 8-byte GPR.  This
/// means that we either have a scalar or we are talking about the high or low
/// part of an up-to-16-byte struct.  This routine picks the best CIR type
/// to represent this, which may be i64 or may be anything else that the
/// backend will pass in a GPR that works better (e.g. i8, %foo*, etc).
///
/// PrefType is an CIR type that corresponds to (part of) the IR type for
/// the source type.  IROffset is an offset in bytes into the CIR type that
/// the 8-byte value references.  PrefType may be null.
///
/// SourceTy is the source-level type for the entire argument.  SourceOffset
/// is an offset into this that we're processing (which is always either 0 or
/// 8).
///
Type X86_64ABIInfo::GetINTEGERTypeAtOffset(Type DestTy, unsigned IROffset,
                                           Type SourceTy,
                                           unsigned SourceOffset) const {
  // If we're dealing with an un-offset CIR type, then it means that we're
  // returning an 8-byte unit starting with it. See if we can safely use it.
  if (IROffset == 0) {
    // Pointers and int64's always fill the 8-byte unit.
    assert(!isa<PointerType>(DestTy) && "Ptrs are NYI");

    // If we have a 1/2/4-byte integer, we can use it only if the rest of the
    // goodness in the source type is just tail padding.  This is allowed to
    // kick in for struct {double,int} on the int, but not on
    // struct{double,int,int} because we wouldn't return the second int.  We
    // have to do this analysis on the source type because we can't depend on
    // unions being lowered a specific way etc.
    if (auto intTy = dyn_cast<IntType>(DestTy)) {
      if (intTy.getWidth() == 8 || intTy.getWidth() == 16 ||
          intTy.getWidth() == 32) {
        unsigned BitWidth = intTy.getWidth();
        if (BitsContainNoUserData(SourceTy, SourceOffset * 8 + BitWidth,
                                  SourceOffset * 8 + 64, getContext()))
          return DestTy;
      }
    }
  }

  if (auto RT = dyn_cast<StructType>(DestTy)) {
    // If this is a struct, recurse into the field at the specified offset.
    const ::cir::StructLayout *SL = getDataLayout().getStructLayout(RT);
    if (IROffset < SL->getSizeInBytes()) {
      unsigned FieldIdx = SL->getElementContainingOffset(IROffset);
      IROffset -= SL->getElementOffset(FieldIdx);

      return GetINTEGERTypeAtOffset(RT.getMembers()[FieldIdx], IROffset,
                                    SourceTy, SourceOffset);
    }
  }

  // Okay, we don't have any better idea of what to pass, so we pass this in
  // an integer register that isn't too big to fit the rest of the struct.
  unsigned TySizeInBytes =
      (unsigned)getContext().getTypeSizeInChars(SourceTy).getQuantity();

  assert(TySizeInBytes != SourceOffset && "Empty field?");

  // It is always safe to classify this as an integer type up to i64 that
  // isn't larger than the structure.
  // FIXME(cir): Perhaps we should have the concept of singless integers in
  // CIR, mostly because coerced types should carry sign. On the other hand,
  // this might not make a difference in practice. For now, we just preserve the
  // sign as is to avoid unecessary bitcasts.
  bool isSigned = false;
  if (auto intTy = dyn_cast<IntType>(SourceTy))
    isSigned = intTy.isSigned();
  return IntType::get(LT.getMLIRContext(),
                      std::min(TySizeInBytes - SourceOffset, 8U) * 8, isSigned);
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

  Type resType = {};
  switch (Lo) {
  case Class::NoClass:
    if (Hi == Class::NoClass)
      return ABIArgInfo::getIgnore();
    break;

  case Class::Integer:
    resType = GetINTEGERTypeAtOffset(RetTy, 0, RetTy, 0);

    // If we have a sign or zero extended integer, make sure to return Extend
    // so that the parameter gets the right LLVM IR attributes.
    if (Hi == Class::NoClass && isa<IntType>(resType)) {
      // NOTE(cir): We skip enum types handling here since CIR represents
      // enums directly as their unerlying integer types. NOTE(cir): For some
      // reason, Clang does not set the coerce type here and delays it to
      // arrangeLLVMFunctionInfo. We do the same to keep parity.
      if (isa<IntType, BoolType>(RetTy) && isPromotableIntegerTypeForABI(RetTy))
        return ABIArgInfo::getExtend(RetTy);
    }
    break;

    // AMD64-ABI 3.2.3p4: Rule 4. If the class is SSE, the next
    // available SSE register of the sequence %xmm0, %xmm1 is used.
  case Class::SSE:
    resType = GetSSETypeAtOffset(RetTy, 0, RetTy, 0);
    break;

  default:
    llvm_unreachable("NYI");
  }

  Type HighPart = {};
  switch (Hi) {

  case Class::NoClass:
    break;

  default:
    llvm_unreachable("NYI");
  }

  // If a high part was specified, merge it together with the low part.  It is
  // known to pass in the high eightbyte of the result.  We do this by forming
  // a first class struct aggregate with the high and low part: {low, high}
  if (HighPart)
    llvm_unreachable("NYI");

  return ABIArgInfo::getDirect(resType);
}

ABIArgInfo X86_64ABIInfo::classifyArgumentType(Type Ty, unsigned freeIntRegs,
                                               unsigned &neededInt,
                                               unsigned &neededSSE,
                                               bool isNamedArg,
                                               bool IsRegCall = false) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  X86_64ABIInfo::Class Lo, Hi;
  classify(Ty, 0, Lo, Hi, isNamedArg, IsRegCall);

  // Check some invariants.
  // FIXME: Enforce these by construction.
  assert((Hi != Class::Memory || Lo == Class::Memory) &&
         "Invalid memory classification.");
  assert((Hi != Class::SSEUp || Lo == Class::SSE) &&
         "Invalid SSEUp classification.");

  neededInt = 0;
  neededSSE = 0;
  Type ResType = {};
  switch (Lo) {
    // AMD64-ABI 3.2.3p3: Rule 2. If the class is INTEGER, the next
    // available register of the sequence %rdi, %rsi, %rdx, %rcx, %r8
    // and %r9 is used.
  case Class::Integer:
    ++neededInt;

    // Pick an 8-byte type based on the preferred type.
    ResType = GetINTEGERTypeAtOffset(Ty, 0, Ty, 0);

    // If we have a sign or zero extended integer, make sure to return Extend
    // so that the parameter gets the right LLVM IR attributes.
    if (Hi == Class::NoClass && isa<IntType>(ResType)) {
      // NOTE(cir): We skip enum types handling here since CIR represents
      // enums directly as their unerlying integer types. NOTE(cir): For some
      // reason, Clang does not set the coerce type here and delays it to
      // arrangeLLVMFunctionInfo. We do the same to keep parity.
      if (isa<IntType, BoolType>(Ty) && isPromotableIntegerTypeForABI(Ty))
        return ABIArgInfo::getExtend(Ty);
    }

    break;

    // AMD64-ABI 3.2.3p3: Rule 3. If the class is SSE, the next
    // available SSE register is used, the registers are taken in the
    // order from %xmm0 to %xmm7.
  case Class::SSE: {
    ResType = GetSSETypeAtOffset(Ty, 0, Ty, 0);
    ++neededSSE;
    break;
  }
  default:
    llvm_unreachable("NYI");
  }

  Type HighPart = {};
  switch (Hi) {
  case Class::NoClass:
    break;
  default:
    llvm_unreachable("NYI");
  }

  if (HighPart)
    llvm_unreachable("NYI");

  return ABIArgInfo::getDirect(ResType);
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
  unsigned FreeIntRegs = IsRegCall ? 11 : 6;
  unsigned FreeSSERegs = IsRegCall ? 16 : 8;
  unsigned NeededInt = 0, NeededSSE = 0, MaxVectorWidth = 0;

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

  unsigned NumRequiredArgs = FI.getNumRequiredArgs();
  // AMD64-ABI 3.2.3p3: Once arguments are classified, the registers
  // get assigned (in left-to-right order) for passing as follows...
  unsigned ArgNo = 0;
  for (LowerFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it, ++ArgNo) {
    bool IsNamedArg = ArgNo < NumRequiredArgs;

    if (IsRegCall && ::cir::MissingFeatures::regCall())
      llvm_unreachable("NYI");
    else
      it->info = classifyArgumentType(it->type, FreeIntRegs, NeededInt,
                                      NeededSSE, IsNamedArg);

    // AMD64-ABI 3.2.3p3: If there are no registers available for any
    // eightbyte of an argument, the whole argument is passed on the
    // stack. If registers have already been assigned for some
    // eightbytes of such an argument, the assignments get reverted.
    if (FreeIntRegs >= NeededInt && FreeSSERegs >= NeededSSE) {
      FreeIntRegs -= NeededInt;
      FreeSSERegs -= NeededSSE;
      if (::cir::MissingFeatures::vectorType())
        llvm_unreachable("NYI");
    } else {
      llvm_unreachable("Indirect results are NYI");
    }
  }
}

X86_64ABIInfo::Class X86_64ABIInfo::merge(Class Accum, Class Field) {
  // AMD64-ABI 3.2.3p2: Rule 4. Each field of an object is
  // classified recursively so that always two fields are
  // considered. The resulting class is calculated according to
  // the classes of the fields in the eightbyte:
  //
  // (a) If both classes are equal, this is the resulting class.
  //
  // (b) If one of the classes is NO_CLASS, the resulting class is
  // the other class.
  //
  // (c) If one of the classes is MEMORY, the result is the MEMORY
  // class.
  //
  // (d) If one of the classes is INTEGER, the result is the
  // INTEGER.
  //
  // (e) If one of the classes is X87, X87UP, COMPLEX_X87 class,
  // MEMORY is used as class.
  //
  // (f) Otherwise class SSE is used.

  // Accum should never be memory (we should have returned) or
  // ComplexX87 (because this cannot be passed in a structure).
  assert((Accum != Class::Memory && Accum != Class::ComplexX87) &&
         "Invalid accumulated classification during merge.");
  if (Accum == Field || Field == Class::NoClass)
    return Accum;
  if (Field == Class::Memory)
    return Class::Memory;
  if (Accum == Class::NoClass)
    return Field;
  if (Accum == Class::Integer || Field == Class::Integer)
    return Class::Integer;
  if (Field == Class::X87 || Field == Class::X87Up ||
      Field == Class::ComplexX87 || Accum == Class::X87 ||
      Accum == Class::X87Up)
    return Class::Memory;
  return Class::SSE;
}

void X86_64ABIInfo::postMerge(unsigned AggregateSize, Class &Lo,
                              Class &Hi) const {
  // AMD64-ABI 3.2.3p2: Rule 5. Then a post merger cleanup is done:
  //
  // (a) If one of the classes is Memory, the whole argument is passed in
  //     memory.
  //
  // (b) If X87UP is not preceded by X87, the whole argument is passed in
  //     memory.
  //
  // (c) If the size of the aggregate exceeds two eightbytes and the first
  //     eightbyte isn't SSE or any other eightbyte isn't SSEUP, the whole
  //     argument is passed in memory. NOTE: This is necessary to keep the
  //     ABI working for processors that don't support the __m256 type.
  //
  // (d) If SSEUP is not preceded by SSE or SSEUP, it is converted to SSE.
  //
  // Some of these are enforced by the merging logic.  Others can arise
  // only with unions; for example:
  //   union { _Complex double; unsigned; }
  //
  // Note that clauses (b) and (c) were added in 0.98.
  //
  if (Hi == Class::Memory)
    Lo = Class::Memory;
  if (Hi == Class::X87Up && Lo != Class::X87 && honorsRevision0_98())
    Lo = Class::Memory;
  if (AggregateSize > 128 && (Lo != Class::SSE || Hi != Class::SSEUp))
    Lo = Class::Memory;
  if (Hi == Class::SSEUp && Lo != Class::SSE)
    Hi = Class::SSE;
}

std::unique_ptr<TargetLoweringInfo>
createX86_64TargetLoweringInfo(LowerModule &LM, X86AVXABILevel AVXLevel) {
  return std::make_unique<X86_64TargetLoweringInfo>(LM.getTypes(), AVXLevel);
}

} // namespace cir
} // namespace mlir
