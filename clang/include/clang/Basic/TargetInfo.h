//===--- TargetInfo.h - Expose information about the target -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the clang::TargetInfo interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_TARGETINFO_H
#define LLVM_CLANG_BASIC_TARGETINFO_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/DataTypes.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/VersionTuple.h"
#include "clang/Basic/Specifiers.h"
#include <cassert>
#include <vector>
#include <string>

namespace llvm {
struct fltSemantics;
}

namespace clang {
class DiagnosticsEngine;
class LangOptions;
class MacroBuilder;
class SourceLocation;
class SourceManager;

namespace Builtin { struct Info; }

/// \brief The types of C++ ABIs for which we can generate code.
enum TargetCXXABI {
  /// The generic ("Itanium") C++ ABI, documented at:
  ///   http://www.codesourcery.com/public/cxx-abi/
  CXXABI_Itanium,

  /// The ARM C++ ABI, based largely on the Itanium ABI but with
  /// significant differences.
  ///    http://infocenter.arm.com
  ///                    /help/topic/com.arm.doc.ihi0041c/IHI0041C_cppabi.pdf
  CXXABI_ARM,

  /// The Visual Studio ABI.  Only scattered official documentation exists.
  CXXABI_Microsoft
};

/// \brief Exposes information about the current target.
///
class TargetInfo : public RefCountedBase<TargetInfo> {
  llvm::IntrusiveRefCntPtr<TargetOptions> TargetOpts;
  llvm::Triple Triple;
protected:
  // Target values set by the ctor of the actual target implementation.  Default
  // values are specified by the TargetInfo constructor.
  bool BigEndian;
  bool TLSSupported;
  bool NoAsmVariants;  // True if {|} are normal characters.
  unsigned char PointerWidth, PointerAlign;
  unsigned char BoolWidth, BoolAlign;
  unsigned char IntWidth, IntAlign;
  unsigned char HalfWidth, HalfAlign;
  unsigned char FloatWidth, FloatAlign;
  unsigned char DoubleWidth, DoubleAlign;
  unsigned char LongDoubleWidth, LongDoubleAlign;
  unsigned char LargeArrayMinWidth, LargeArrayAlign;
  unsigned char LongWidth, LongAlign;
  unsigned char LongLongWidth, LongLongAlign;
  unsigned char SuitableAlign;
  unsigned char MaxAtomicPromoteWidth, MaxAtomicInlineWidth;
  unsigned short MaxVectorAlign;
  const char *DescriptionString;
  const char *UserLabelPrefix;
  const char *MCountName;
  const llvm::fltSemantics *HalfFormat, *FloatFormat, *DoubleFormat,
    *LongDoubleFormat;
  unsigned char RegParmMax, SSERegParmMax;
  TargetCXXABI CXXABI;
  const LangAS::Map *AddrSpaceMap;

  mutable StringRef PlatformName;
  mutable VersionTuple PlatformMinVersion;

  unsigned HasAlignMac68kSupport : 1;
  unsigned RealTypeUsesObjCFPRet : 3;
  unsigned ComplexLongDoubleUsesFP2Ret : 1;

  // TargetInfo Constructor.  Default initializes all fields.
  TargetInfo(const std::string &T);

public:
  /// \brief Construct a target for the given options.
  ///
  /// \param Opts - The options to use to initialize the target. The target may
  /// modify the options to canonicalize the target feature information to match
  /// what the backend expects.
  static TargetInfo* CreateTargetInfo(DiagnosticsEngine &Diags,
                                      TargetOptions &Opts);

  virtual ~TargetInfo();

  /// \brief Retrieve the target options.
  TargetOptions &getTargetOpts() const { 
    assert(TargetOpts && "Missing target options");
    return *TargetOpts; 
  }

  void setTargetOpts(TargetOptions &TargetOpts) {
    this->TargetOpts = &TargetOpts;
  }

  ///===---- Target Data Type Query Methods -------------------------------===//
  enum IntType {
    NoInt = 0,
    SignedShort,
    UnsignedShort,
    SignedInt,
    UnsignedInt,
    SignedLong,
    UnsignedLong,
    SignedLongLong,
    UnsignedLongLong
  };

  enum RealType {
    Float = 0,
    Double,
    LongDouble
  };

  /// \brief The different kinds of __builtin_va_list types defined by
  /// the target implementation.
  enum BuiltinVaListKind {
    /// typedef char* __builtin_va_list;
    CharPtrBuiltinVaList = 0,

    /// typedef void* __builtin_va_list;
    VoidPtrBuiltinVaList,

    /// __builtin_va_list as defined by the PNaCl ABI:
    /// http://www.chromium.org/nativeclient/pnacl/bitcode-abi#TOC-Machine-Types
    PNaClABIBuiltinVaList,

    /// __builtin_va_list as defined by the Power ABI:
    /// https://www.power.org
    ///        /resources/downloads/Power-Arch-32-bit-ABI-supp-1.0-Embedded.pdf
    PowerABIBuiltinVaList,

    /// __builtin_va_list as defined by the x86-64 ABI:
    /// http://www.x86-64.org/documentation/abi.pdf
    X86_64ABIBuiltinVaList,

    /// __builtin_va_list as defined by ARM AAPCS ABI
    /// http://infocenter.arm.com
    //        /help/topic/com.arm.doc.ihi0042d/IHI0042D_aapcs.pdf
    AAPCSABIBuiltinVaList
  };

protected:
  IntType SizeType, IntMaxType, UIntMaxType, PtrDiffType, IntPtrType, WCharType,
          WIntType, Char16Type, Char32Type, Int64Type, SigAtomicType;

  /// \brief Whether Objective-C's built-in boolean type should be signed char.
  ///
  /// Otherwise, when this flag is not set, the normal built-in boolean type is
  /// used.
  unsigned UseSignedCharForObjCBool : 1;

  /// Control whether the alignment of bit-field types is respected when laying
  /// out structures. If true, then the alignment of the bit-field type will be
  /// used to (a) impact the alignment of the containing structure, and (b)
  /// ensure that the individual bit-field will not straddle an alignment
  /// boundary.
  unsigned UseBitFieldTypeAlignment : 1;

  /// \brief Whether zero length bitfields (e.g., int : 0;) force alignment of
  /// the next bitfield.
  ///
  /// If the alignment of the zero length bitfield is greater than the member
  /// that follows it, `bar', `bar' will be aligned as the type of the
  /// zero-length bitfield.
  unsigned UseZeroLengthBitfieldAlignment : 1;

  /// If non-zero, specifies a fixed alignment value for bitfields that follow
  /// zero length bitfield, regardless of the zero length bitfield type.
  unsigned ZeroLengthBitfieldBoundary;

public:
  IntType getSizeType() const { return SizeType; }
  IntType getIntMaxType() const { return IntMaxType; }
  IntType getUIntMaxType() const { return UIntMaxType; }
  IntType getPtrDiffType(unsigned AddrSpace) const {
    return AddrSpace == 0 ? PtrDiffType : getPtrDiffTypeV(AddrSpace);
  }
  IntType getIntPtrType() const { return IntPtrType; }
  IntType getWCharType() const { return WCharType; }
  IntType getWIntType() const { return WIntType; }
  IntType getChar16Type() const { return Char16Type; }
  IntType getChar32Type() const { return Char32Type; }
  IntType getInt64Type() const { return Int64Type; }
  IntType getSigAtomicType() const { return SigAtomicType; }


  /// \brief Return the width (in bits) of the specified integer type enum.
  ///
  /// For example, SignedInt -> getIntWidth().
  unsigned getTypeWidth(IntType T) const;

  /// \brief Return the alignment (in bits) of the specified integer type enum.
  ///
  /// For example, SignedInt -> getIntAlign().
  unsigned getTypeAlign(IntType T) const;

  /// \brief Returns true if the type is signed; false otherwise.
  static bool isTypeSigned(IntType T);

  /// \brief Return the width of pointers on this target, for the
  /// specified address space.
  uint64_t getPointerWidth(unsigned AddrSpace) const {
    return AddrSpace == 0 ? PointerWidth : getPointerWidthV(AddrSpace);
  }
  uint64_t getPointerAlign(unsigned AddrSpace) const {
    return AddrSpace == 0 ? PointerAlign : getPointerAlignV(AddrSpace);
  }

  /// \brief Return the size of '_Bool' and C++ 'bool' for this target, in bits.
  unsigned getBoolWidth() const { return BoolWidth; }

  /// \brief Return the alignment of '_Bool' and C++ 'bool' for this target.
  unsigned getBoolAlign() const { return BoolAlign; }

  unsigned getCharWidth() const { return 8; } // FIXME
  unsigned getCharAlign() const { return 8; } // FIXME

  /// \brief Return the size of 'signed short' and 'unsigned short' for this
  /// target, in bits.
  unsigned getShortWidth() const { return 16; } // FIXME

  /// \brief Return the alignment of 'signed short' and 'unsigned short' for
  /// this target.
  unsigned getShortAlign() const { return 16; } // FIXME

  /// getIntWidth/Align - Return the size of 'signed int' and 'unsigned int' for
  /// this target, in bits.
  unsigned getIntWidth() const { return IntWidth; }
  unsigned getIntAlign() const { return IntAlign; }

  /// getLongWidth/Align - Return the size of 'signed long' and 'unsigned long'
  /// for this target, in bits.
  unsigned getLongWidth() const { return LongWidth; }
  unsigned getLongAlign() const { return LongAlign; }

  /// getLongLongWidth/Align - Return the size of 'signed long long' and
  /// 'unsigned long long' for this target, in bits.
  unsigned getLongLongWidth() const { return LongLongWidth; }
  unsigned getLongLongAlign() const { return LongLongAlign; }

  /// \brief Return the alignment that is suitable for storing any
  /// object with a fundamental alignment requirement.
  unsigned getSuitableAlign() const { return SuitableAlign; }

  /// getWCharWidth/Align - Return the size of 'wchar_t' for this target, in
  /// bits.
  unsigned getWCharWidth() const { return getTypeWidth(WCharType); }
  unsigned getWCharAlign() const { return getTypeAlign(WCharType); }

  /// getChar16Width/Align - Return the size of 'char16_t' for this target, in
  /// bits.
  unsigned getChar16Width() const { return getTypeWidth(Char16Type); }
  unsigned getChar16Align() const { return getTypeAlign(Char16Type); }

  /// getChar32Width/Align - Return the size of 'char32_t' for this target, in
  /// bits.
  unsigned getChar32Width() const { return getTypeWidth(Char32Type); }
  unsigned getChar32Align() const { return getTypeAlign(Char32Type); }

  /// getHalfWidth/Align/Format - Return the size/align/format of 'half'.
  unsigned getHalfWidth() const { return HalfWidth; }
  unsigned getHalfAlign() const { return HalfAlign; }
  const llvm::fltSemantics &getHalfFormat() const { return *HalfFormat; }

  /// getFloatWidth/Align/Format - Return the size/align/format of 'float'.
  unsigned getFloatWidth() const { return FloatWidth; }
  unsigned getFloatAlign() const { return FloatAlign; }
  const llvm::fltSemantics &getFloatFormat() const { return *FloatFormat; }

  /// getDoubleWidth/Align/Format - Return the size/align/format of 'double'.
  unsigned getDoubleWidth() const { return DoubleWidth; }
  unsigned getDoubleAlign() const { return DoubleAlign; }
  const llvm::fltSemantics &getDoubleFormat() const { return *DoubleFormat; }

  /// getLongDoubleWidth/Align/Format - Return the size/align/format of 'long
  /// double'.
  unsigned getLongDoubleWidth() const { return LongDoubleWidth; }
  unsigned getLongDoubleAlign() const { return LongDoubleAlign; }
  const llvm::fltSemantics &getLongDoubleFormat() const {
    return *LongDoubleFormat;
  }

  /// \brief Return the value for the C99 FLT_EVAL_METHOD macro.
  virtual unsigned getFloatEvalMethod() const { return 0; }

  // getLargeArrayMinWidth/Align - Return the minimum array size that is
  // 'large' and its alignment.
  unsigned getLargeArrayMinWidth() const { return LargeArrayMinWidth; }
  unsigned getLargeArrayAlign() const { return LargeArrayAlign; }

  /// \brief Return the maximum width lock-free atomic operation which will
  /// ever be supported for the given target
  unsigned getMaxAtomicPromoteWidth() const { return MaxAtomicPromoteWidth; }
  /// \brief Return the maximum width lock-free atomic operation which can be
  /// inlined given the supported features of the given target.
  unsigned getMaxAtomicInlineWidth() const { return MaxAtomicInlineWidth; }

  /// \brief Return the maximum vector alignment supported for the given target.
  unsigned getMaxVectorAlign() const { return MaxVectorAlign; }

  /// \brief Return the size of intmax_t and uintmax_t for this target, in bits.
  unsigned getIntMaxTWidth() const {
    return getTypeWidth(IntMaxType);
  }

  /// \brief Return the "preferred" register width on this target.
  uint64_t getRegisterWidth() const {
    // Currently we assume the register width on the target matches the pointer
    // width, we can introduce a new variable for this if/when some target wants
    // it.
    return LongWidth; 
  }

  /// \brief Returns the default value of the __USER_LABEL_PREFIX__ macro,
  /// which is the prefix given to user symbols by default.
  ///
  /// On most platforms this is "_", but it is "" on some, and "." on others.
  const char *getUserLabelPrefix() const {
    return UserLabelPrefix;
  }

  /// \brief Returns the name of the mcount instrumentation function.
  const char *getMCountName() const {
    return MCountName;
  }

  /// \brief Check if the Objective-C built-in boolean type should be signed
  /// char.
  ///
  /// Otherwise, if this returns false, the normal built-in boolean type
  /// should also be used for Objective-C.
  bool useSignedCharForObjCBool() const {
    return UseSignedCharForObjCBool;
  }
  void noSignedCharForObjCBool() {
    UseSignedCharForObjCBool = false;
  }

  /// \brief Check whether the alignment of bit-field types is respected
  /// when laying out structures.
  bool useBitFieldTypeAlignment() const {
    return UseBitFieldTypeAlignment;
  }

  /// \brief Check whether zero length bitfields should force alignment of
  /// the next member.
  bool useZeroLengthBitfieldAlignment() const {
    return UseZeroLengthBitfieldAlignment;
  }

  /// \brief Get the fixed alignment value in bits for a member that follows
  /// a zero length bitfield.
  unsigned getZeroLengthBitfieldBoundary() const {
    return ZeroLengthBitfieldBoundary;
  }

  /// \brief Check whether this target support '\#pragma options align=mac68k'.
  bool hasAlignMac68kSupport() const {
    return HasAlignMac68kSupport;
  }

  /// \brief Return the user string for the specified integer type enum.
  ///
  /// For example, SignedShort -> "short".
  static const char *getTypeName(IntType T);

  /// \brief Return the constant suffix for the specified integer type enum.
  ///
  /// For example, SignedLong -> "L".
  static const char *getTypeConstantSuffix(IntType T);

  /// \brief Check whether the given real type should use the "fpret" flavor of
  /// Objective-C message passing on this target.
  bool useObjCFPRetForRealType(RealType T) const {
    return RealTypeUsesObjCFPRet & (1 << T);
  }

  /// \brief Check whether _Complex long double should use the "fp2ret" flavor
  /// of Objective-C message passing on this target.
  bool useObjCFP2RetForComplexLongDouble() const {
    return ComplexLongDoubleUsesFP2Ret;
  }

  ///===---- Other target property query methods --------------------------===//

  /// \brief Appends the target-specific \#define values for this
  /// target set to the specified buffer.
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const = 0;


  /// Return information about target-specific builtins for
  /// the current primary target, and info about which builtins are non-portable
  /// across the current set of primary and secondary targets.
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const = 0;

  /// The __builtin_clz* and __builtin_ctz* built-in
  /// functions are specified to have undefined results for zero inputs, but
  /// on targets that support these operations in a way that provides
  /// well-defined results for zero without loss of performance, it is a good
  /// idea to avoid optimizing based on that undef behavior.
  virtual bool isCLZForZeroUndef() const { return true; }

  /// \brief Returns the kind of __builtin_va_list type that should be used
  /// with this target.
  virtual BuiltinVaListKind getBuiltinVaListKind() const = 0;

  /// \brief Returns whether the passed in string is a valid clobber in an
  /// inline asm statement.
  ///
  /// This is used by Sema.
  bool isValidClobber(StringRef Name) const;

  /// \brief Returns whether the passed in string is a valid register name
  /// according to GCC.
  ///
  /// This is used by Sema for inline asm statements.
  bool isValidGCCRegisterName(StringRef Name) const;

  /// \brief Returns the "normalized" GCC register name.
  ///
  /// For example, on x86 it will return "ax" when "eax" is passed in.
  StringRef getNormalizedGCCRegisterName(StringRef Name) const;

  struct ConstraintInfo {
    enum {
      CI_None = 0x00,
      CI_AllowsMemory = 0x01,
      CI_AllowsRegister = 0x02,
      CI_ReadWrite = 0x04,       // "+r" output constraint (read and write).
      CI_HasMatchingInput = 0x08 // This output operand has a matching input.
    };
    unsigned Flags;
    int TiedOperand;

    std::string ConstraintStr;  // constraint: "=rm"
    std::string Name;           // Operand name: [foo] with no []'s.
  public:
    ConstraintInfo(StringRef ConstraintStr, StringRef Name)
      : Flags(0), TiedOperand(-1), ConstraintStr(ConstraintStr.str()),
      Name(Name.str()) {}

    const std::string &getConstraintStr() const { return ConstraintStr; }
    const std::string &getName() const { return Name; }
    bool isReadWrite() const { return (Flags & CI_ReadWrite) != 0; }
    bool allowsRegister() const { return (Flags & CI_AllowsRegister) != 0; }
    bool allowsMemory() const { return (Flags & CI_AllowsMemory) != 0; }

    /// \brief Return true if this output operand has a matching
    /// (tied) input operand.
    bool hasMatchingInput() const { return (Flags & CI_HasMatchingInput) != 0; }

    /// \brief Return true if this input operand is a matching
    /// constraint that ties it to an output operand.
    ///
    /// If this returns true then getTiedOperand will indicate which output
    /// operand this is tied to.
    bool hasTiedOperand() const { return TiedOperand != -1; }
    unsigned getTiedOperand() const {
      assert(hasTiedOperand() && "Has no tied operand!");
      return (unsigned)TiedOperand;
    }

    void setIsReadWrite() { Flags |= CI_ReadWrite; }
    void setAllowsMemory() { Flags |= CI_AllowsMemory; }
    void setAllowsRegister() { Flags |= CI_AllowsRegister; }
    void setHasMatchingInput() { Flags |= CI_HasMatchingInput; }

    /// \brief Indicate that this is an input operand that is tied to
    /// the specified output operand. 
    ///
    /// Copy over the various constraint information from the output.
    void setTiedOperand(unsigned N, ConstraintInfo &Output) {
      Output.setHasMatchingInput();
      Flags = Output.Flags;
      TiedOperand = N;
      // Don't copy Name or constraint string.
    }
  };

  // validateOutputConstraint, validateInputConstraint - Checks that
  // a constraint is valid and provides information about it.
  // FIXME: These should return a real error instead of just true/false.
  bool validateOutputConstraint(ConstraintInfo &Info) const;
  bool validateInputConstraint(ConstraintInfo *OutputConstraints,
                               unsigned NumOutputs,
                               ConstraintInfo &info) const;
  virtual bool validateConstraintModifier(StringRef /*Constraint*/,
                                          const char /*Modifier*/,
                                          unsigned /*Size*/) const {
    return true;
  }
  bool resolveSymbolicName(const char *&Name,
                           ConstraintInfo *OutputConstraints,
                           unsigned NumOutputs, unsigned &Index) const;

  // Constraint parm will be left pointing at the last character of
  // the constraint.  In practice, it won't be changed unless the
  // constraint is longer than one character.
  virtual std::string convertConstraint(const char *&Constraint) const {
    // 'p' defaults to 'r', but can be overridden by targets.
    if (*Constraint == 'p')
      return std::string("r");
    return std::string(1, *Constraint);
  }

  /// \brief Returns a string of target-specific clobbers, in LLVM format.
  virtual const char *getClobbers() const = 0;


  /// \brief Returns the target triple of the primary target.
  const llvm::Triple &getTriple() const {
    return Triple;
  }

  const char *getTargetDescription() const {
    return DescriptionString;
  }

  struct GCCRegAlias {
    const char * const Aliases[5];
    const char * const Register;
  };

  struct AddlRegName {
    const char * const Names[5];
    const unsigned RegNum;
  };

  /// \brief Does this target support "protected" visibility?
  ///
  /// Any target which dynamic libraries will naturally support
  /// something like "default" (meaning that the symbol is visible
  /// outside this shared object) and "hidden" (meaning that it isn't)
  /// visibilities, but "protected" is really an ELF-specific concept
  /// with weird semantics designed around the convenience of dynamic
  /// linker implementations.  Which is not to suggest that there's
  /// consistent target-independent semantics for "default" visibility
  /// either; the entire thing is pretty badly mangled.
  virtual bool hasProtectedVisibility() const { return true; }

  virtual bool useGlobalsForAutomaticVariables() const { return false; }

  /// \brief Return the section to use for CFString literals, or 0 if no
  /// special section is used.
  virtual const char *getCFStringSection() const {
    return "__DATA,__cfstring";
  }

  /// \brief Return the section to use for NSString literals, or 0 if no
  /// special section is used.
  virtual const char *getNSStringSection() const {
    return "__OBJC,__cstring_object,regular,no_dead_strip";
  }

  /// \brief Return the section to use for NSString literals, or 0 if no
  /// special section is used (NonFragile ABI).
  virtual const char *getNSStringNonFragileABISection() const {
    return "__DATA, __objc_stringobj, regular, no_dead_strip";
  }

  /// \brief An optional hook that targets can implement to perform semantic
  /// checking on attribute((section("foo"))) specifiers.
  ///
  /// In this case, "foo" is passed in to be checked.  If the section
  /// specifier is invalid, the backend should return a non-empty string
  /// that indicates the problem.
  ///
  /// This hook is a simple quality of implementation feature to catch errors
  /// and give good diagnostics in cases when the assembler or code generator
  /// would otherwise reject the section specifier.
  ///
  virtual std::string isValidSectionSpecifier(StringRef SR) const {
    return "";
  }

  /// \brief Set forced language options.
  ///
  /// Apply changes to the target information with respect to certain
  /// language options which change the target configuration.
  virtual void setForcedLangOptions(LangOptions &Opts);

  /// \brief Get the default set of target features for the CPU;
  /// this should include all legal feature strings on the target.
  virtual void getDefaultFeatures(llvm::StringMap<bool> &Features) const {
  }

  /// \brief Get the ABI currently in use.
  virtual const char *getABI() const {
    return "";
  }

  /// \brief Get the C++ ABI currently in use.
  virtual TargetCXXABI getCXXABI() const {
    return CXXABI;
  }

  /// \brief Target the specified CPU.
  ///
  /// \return  False on error (invalid CPU name).
  virtual bool setCPU(const std::string &Name) {
    return false;
  }

  /// \brief Use the specified ABI.
  ///
  /// \return False on error (invalid ABI name).
  virtual bool setABI(const std::string &Name) {
    return false;
  }

  /// \brief Use this specified C++ ABI.
  ///
  /// \return False on error (invalid C++ ABI name).
  bool setCXXABI(const std::string &Name) {
    static const TargetCXXABI Unknown = static_cast<TargetCXXABI>(-1);
    TargetCXXABI ABI = llvm::StringSwitch<TargetCXXABI>(Name)
      .Case("arm", CXXABI_ARM)
      .Case("itanium", CXXABI_Itanium)
      .Case("microsoft", CXXABI_Microsoft)
      .Default(Unknown);
    if (ABI == Unknown) return false;
    return setCXXABI(ABI);
  }

  /// \brief Set the C++ ABI to be used by this implementation.
  ///
  /// \return False on error (ABI not valid on this target)
  virtual bool setCXXABI(TargetCXXABI ABI) {
    CXXABI = ABI;
    return true;
  }

  /// \brief Enable or disable a specific target feature;
  /// the feature name must be valid.
  ///
  /// \return False on error (invalid feature name).
  virtual bool setFeatureEnabled(llvm::StringMap<bool> &Features,
                                 StringRef Name,
                                 bool Enabled) const {
    return false;
  }

  /// \brief Perform initialization based on the user configured
  /// set of features (e.g., +sse4).
  ///
  /// The list is guaranteed to have at most one entry per feature.
  ///
  /// The target may modify the features list, to change which options are
  /// passed onwards to the backend.
  virtual void HandleTargetFeatures(std::vector<std::string> &Features) {
  }

  /// \brief Determine whether the given target has the given feature.
  virtual bool hasFeature(StringRef Feature) const {
    return false;
  }
  
  // \brief Returns maximal number of args passed in registers.
  unsigned getRegParmMax() const {
    assert(RegParmMax < 7 && "RegParmMax value is larger than AST can handle");
    return RegParmMax;
  }

  /// \brief Whether the target supports thread-local storage.
  bool isTLSSupported() const {
    return TLSSupported;
  }

  /// \brief Return true if {|} are normal characters in the asm string.
  ///
  /// If this returns false (the default), then {abc|xyz} is syntax
  /// that says that when compiling for asm variant #0, "abc" should be
  /// generated, but when compiling for asm variant #1, "xyz" should be
  /// generated.
  bool hasNoAsmVariants() const {
    return NoAsmVariants;
  }

  /// \brief Return the register number that __builtin_eh_return_regno would
  /// return with the specified argument.
  virtual int getEHDataRegisterNumber(unsigned RegNo) const {
    return -1;
  }

  /// \brief Return the section to use for C++ static initialization functions.
  virtual const char *getStaticInitSectionSpecifier() const {
    return 0;
  }

  const LangAS::Map &getAddressSpaceMap() const {
    return *AddrSpaceMap;
  }

  /// \brief Retrieve the name of the platform as it is used in the
  /// availability attribute.
  StringRef getPlatformName() const { return PlatformName; }

  /// \brief Retrieve the minimum desired version of the platform, to
  /// which the program should be compiled.
  VersionTuple getPlatformMinVersion() const { return PlatformMinVersion; }

  bool isBigEndian() const { return BigEndian; }

  /// \brief Gets the default calling convention for the given target and
  /// declaration context.
  virtual CallingConv getDefaultCallingConv() const {
    // Not all targets will specify an explicit calling convention that we can
    // express.  This will always do the right thing, even though it's not
    // an explicit calling convention.
    return CC_Default;
  }

  enum CallingConvCheckResult {
    CCCR_OK,
    CCCR_Warning
  };

  /// \brief Determines whether a given calling convention is valid for the
  /// target. A calling convention can either be accepted, produce a warning 
  /// and be substituted with the default calling convention, or (someday)
  /// produce an error (such as using thiscall on a non-instance function).
  virtual CallingConvCheckResult checkCallingConvention(CallingConv CC) const {
    switch (CC) {
      default:
        return CCCR_Warning;
      case CC_C:
      case CC_Default:
        return CCCR_OK;
    }
  }

protected:
  virtual uint64_t getPointerWidthV(unsigned AddrSpace) const {
    return PointerWidth;
  }
  virtual uint64_t getPointerAlignV(unsigned AddrSpace) const {
    return PointerAlign;
  }
  virtual enum IntType getPtrDiffTypeV(unsigned AddrSpace) const {
    return PtrDiffType;
  }
  virtual void getGCCRegNames(const char * const *&Names,
                              unsigned &NumNames) const = 0;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const = 0;
  virtual void getGCCAddlRegNames(const AddlRegName *&Addl,
				  unsigned &NumAddl) const {
    Addl = 0;
    NumAddl = 0;
  }
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &info) const= 0;
};

}  // end namespace clang

#endif
