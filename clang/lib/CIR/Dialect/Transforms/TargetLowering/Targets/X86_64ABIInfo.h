#include "ABIInfo.h"
#include "clang/CIR/Target/x86.h"

namespace cir {
class X86_64ABIInfo : public cir::ABIInfo {
  using Class = cir::X86ArgClass;

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
  void classify(mlir::Type T, uint64_t OffsetBase, Class &Lo, Class &Hi,
                bool isNamedArg, bool IsRegCall = false) const;

  mlir::Type GetSSETypeAtOffset(mlir::Type IRType, unsigned IROffset,
                                mlir::Type SourceTy,
                                unsigned SourceOffset) const;

  mlir::Type GetINTEGERTypeAtOffset(mlir::Type DestTy, unsigned IROffset,
                                    mlir::Type SourceTy,
                                    unsigned SourceOffset) const;

  /// The 0.98 ABI revision clarified a lot of ambiguities,
  /// unfortunately in ways that were not always consistent with
  /// certain previous compilers.  In particular, platforms which
  /// required strict binary compatibility with older versions of GCC
  /// may need to exempt themselves.
  bool honorsRevision0_98() const {
    return !getTarget().getTriple().isOSDarwin();
  }

  ::cir::X86AVXABILevel AVXLevel;

public:
  X86_64ABIInfo(LowerTypes &CGT, cir::X86AVXABILevel AVXLevel)
      : ABIInfo(CGT), AVXLevel(AVXLevel) {}

  cir::ABIArgInfo classifyReturnType(mlir::Type RetTy) const;

  cir::ABIArgInfo classifyArgumentType(mlir::Type Ty, unsigned freeIntRegs,
                                       unsigned &neededInt, unsigned &neededSSE,
                                       bool isNamedArg, bool IsRegCall) const;

  void computeInfo(LowerFunctionInfo &FI) const override;
};

} // namespace cir
