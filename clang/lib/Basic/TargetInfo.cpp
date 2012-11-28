//===--- TargetInfo.cpp - Information about Target machine ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the TargetInfo and TargetInfoImpl interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
#include <cstdlib>
using namespace clang;

static const LangAS::Map DefaultAddrSpaceMap = { 0 };

// TargetInfo Constructor.
TargetInfo::TargetInfo(const std::string &T) : TargetOpts(), Triple(T)
{
  // Set defaults.  Defaults are set for a 32-bit RISC platform, like PPC or
  // SPARC.  These should be overridden by concrete targets as needed.
  BigEndian = true;
  TLSSupported = true;
  NoAsmVariants = false;
  PointerWidth = PointerAlign = 32;
  BoolWidth = BoolAlign = 8;
  IntWidth = IntAlign = 32;
  LongWidth = LongAlign = 32;
  LongLongWidth = LongLongAlign = 64;
  SuitableAlign = 64;
  HalfWidth = 16;
  HalfAlign = 16;
  FloatWidth = 32;
  FloatAlign = 32;
  DoubleWidth = 64;
  DoubleAlign = 64;
  LongDoubleWidth = 64;
  LongDoubleAlign = 64;
  LargeArrayMinWidth = 0;
  LargeArrayAlign = 0;
  MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 0;
  MaxVectorAlign = 0;
  SizeType = UnsignedLong;
  PtrDiffType = SignedLong;
  IntMaxType = SignedLongLong;
  UIntMaxType = UnsignedLongLong;
  IntPtrType = SignedLong;
  WCharType = SignedInt;
  WIntType = SignedInt;
  Char16Type = UnsignedShort;
  Char32Type = UnsignedInt;
  Int64Type = SignedLongLong;
  SigAtomicType = SignedInt;
  ProcessIDType = SignedInt;
  UseSignedCharForObjCBool = true;
  UseBitFieldTypeAlignment = true;
  UseZeroLengthBitfieldAlignment = false;
  ZeroLengthBitfieldBoundary = 0;
  HalfFormat = &llvm::APFloat::IEEEhalf;
  FloatFormat = &llvm::APFloat::IEEEsingle;
  DoubleFormat = &llvm::APFloat::IEEEdouble;
  LongDoubleFormat = &llvm::APFloat::IEEEdouble;
  DescriptionString = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                      "i64:64:64-f32:32:32-f64:64:64-n32";
  UserLabelPrefix = "_";
  MCountName = "mcount";
  RegParmMax = 0;
  SSERegParmMax = 0;
  HasAlignMac68kSupport = false;

  // Default to no types using fpret.
  RealTypeUsesObjCFPRet = 0;

  // Default to not using fp2ret for __Complex long double
  ComplexLongDoubleUsesFP2Ret = false;

  // Default to using the Itanium ABI.
  CXXABI = CXXABI_Itanium;

  // Default to an empty address space map.
  AddrSpaceMap = &DefaultAddrSpaceMap;

  // Default to an unknown platform name.
  PlatformName = "unknown";
  PlatformMinVersion = VersionTuple();
}

// Out of line virtual dtor for TargetInfo.
TargetInfo::~TargetInfo() {}

/// getTypeName - Return the user string for the specified integer type enum.
/// For example, SignedShort -> "short".
const char *TargetInfo::getTypeName(IntType T) {
  switch (T) {
  default: llvm_unreachable("not an integer!");
  case SignedShort:      return "short";
  case UnsignedShort:    return "unsigned short";
  case SignedInt:        return "int";
  case UnsignedInt:      return "unsigned int";
  case SignedLong:       return "long int";
  case UnsignedLong:     return "long unsigned int";
  case SignedLongLong:   return "long long int";
  case UnsignedLongLong: return "long long unsigned int";
  }
}

/// getTypeConstantSuffix - Return the constant suffix for the specified
/// integer type enum. For example, SignedLong -> "L".
const char *TargetInfo::getTypeConstantSuffix(IntType T) {
  switch (T) {
  default: llvm_unreachable("not an integer!");
  case SignedShort:
  case SignedInt:        return "";
  case SignedLong:       return "L";
  case SignedLongLong:   return "LL";
  case UnsignedShort:
  case UnsignedInt:      return "U";
  case UnsignedLong:     return "UL";
  case UnsignedLongLong: return "ULL";
  }
}

/// getTypeWidth - Return the width (in bits) of the specified integer type
/// enum. For example, SignedInt -> getIntWidth().
unsigned TargetInfo::getTypeWidth(IntType T) const {
  switch (T) {
  default: llvm_unreachable("not an integer!");
  case SignedShort:
  case UnsignedShort:    return getShortWidth();
  case SignedInt:
  case UnsignedInt:      return getIntWidth();
  case SignedLong:
  case UnsignedLong:     return getLongWidth();
  case SignedLongLong:
  case UnsignedLongLong: return getLongLongWidth();
  };
}

/// getTypeAlign - Return the alignment (in bits) of the specified integer type
/// enum. For example, SignedInt -> getIntAlign().
unsigned TargetInfo::getTypeAlign(IntType T) const {
  switch (T) {
  default: llvm_unreachable("not an integer!");
  case SignedShort:
  case UnsignedShort:    return getShortAlign();
  case SignedInt:
  case UnsignedInt:      return getIntAlign();
  case SignedLong:
  case UnsignedLong:     return getLongAlign();
  case SignedLongLong:
  case UnsignedLongLong: return getLongLongAlign();
  };
}

/// isTypeSigned - Return whether an integer types is signed. Returns true if
/// the type is signed; false otherwise.
bool TargetInfo::isTypeSigned(IntType T) {
  switch (T) {
  default: llvm_unreachable("not an integer!");
  case SignedShort:
  case SignedInt:
  case SignedLong:
  case SignedLongLong:
    return true;
  case UnsignedShort:
  case UnsignedInt:
  case UnsignedLong:
  case UnsignedLongLong:
    return false;
  };
}

/// setForcedLangOptions - Set forced language options.
/// Apply changes to the target information with respect to certain
/// language options which change the target configuration.
void TargetInfo::setForcedLangOptions(LangOptions &Opts) {
  if (Opts.NoBitFieldTypeAlign)
    UseBitFieldTypeAlignment = false;
  if (Opts.ShortWChar)
    WCharType = UnsignedShort;
}

//===----------------------------------------------------------------------===//


static StringRef removeGCCRegisterPrefix(StringRef Name) {
  if (Name[0] == '%' || Name[0] == '#')
    Name = Name.substr(1);

  return Name;
}

/// isValidClobber - Returns whether the passed in string is
/// a valid clobber in an inline asm statement. This is used by
/// Sema.
bool TargetInfo::isValidClobber(StringRef Name) const {
  return (isValidGCCRegisterName(Name) ||
	  Name == "memory" || Name == "cc");
}

/// isValidGCCRegisterName - Returns whether the passed in string
/// is a valid register name according to GCC. This is used by Sema for
/// inline asm statements.
bool TargetInfo::isValidGCCRegisterName(StringRef Name) const {
  if (Name.empty())
    return false;

  const char * const *Names;
  unsigned NumNames;

  // Get rid of any register prefix.
  Name = removeGCCRegisterPrefix(Name);

  getGCCRegNames(Names, NumNames);

  // If we have a number it maps to an entry in the register name array.
  if (isdigit(Name[0])) {
    int n;
    if (!Name.getAsInteger(0, n))
      return n >= 0 && (unsigned)n < NumNames;
  }

  // Check register names.
  for (unsigned i = 0; i < NumNames; i++) {
    if (Name == Names[i])
      return true;
  }

  // Check any additional names that we have.
  const AddlRegName *AddlNames;
  unsigned NumAddlNames;
  getGCCAddlRegNames(AddlNames, NumAddlNames);
  for (unsigned i = 0; i < NumAddlNames; i++)
    for (unsigned j = 0; j < llvm::array_lengthof(AddlNames[i].Names); j++) {
      if (!AddlNames[i].Names[j])
	break;
      // Make sure the register that the additional name is for is within
      // the bounds of the register names from above.
      if (AddlNames[i].Names[j] == Name && AddlNames[i].RegNum < NumNames)
	return true;
  }

  // Now check aliases.
  const GCCRegAlias *Aliases;
  unsigned NumAliases;

  getGCCRegAliases(Aliases, NumAliases);
  for (unsigned i = 0; i < NumAliases; i++) {
    for (unsigned j = 0 ; j < llvm::array_lengthof(Aliases[i].Aliases); j++) {
      if (!Aliases[i].Aliases[j])
        break;
      if (Aliases[i].Aliases[j] == Name)
        return true;
    }
  }

  return false;
}

StringRef
TargetInfo::getNormalizedGCCRegisterName(StringRef Name) const {
  assert(isValidGCCRegisterName(Name) && "Invalid register passed in");

  // Get rid of any register prefix.
  Name = removeGCCRegisterPrefix(Name);

  const char * const *Names;
  unsigned NumNames;

  getGCCRegNames(Names, NumNames);

  // First, check if we have a number.
  if (isdigit(Name[0])) {
    int n;
    if (!Name.getAsInteger(0, n)) {
      assert(n >= 0 && (unsigned)n < NumNames &&
             "Out of bounds register number!");
      return Names[n];
    }
  }

  // Check any additional names that we have.
  const AddlRegName *AddlNames;
  unsigned NumAddlNames;
  getGCCAddlRegNames(AddlNames, NumAddlNames);
  for (unsigned i = 0; i < NumAddlNames; i++)
    for (unsigned j = 0; j < llvm::array_lengthof(AddlNames[i].Names); j++) {
      if (!AddlNames[i].Names[j])
	break;
      // Make sure the register that the additional name is for is within
      // the bounds of the register names from above.
      if (AddlNames[i].Names[j] == Name && AddlNames[i].RegNum < NumNames)
	return Name;
    }

  // Now check aliases.
  const GCCRegAlias *Aliases;
  unsigned NumAliases;

  getGCCRegAliases(Aliases, NumAliases);
  for (unsigned i = 0; i < NumAliases; i++) {
    for (unsigned j = 0 ; j < llvm::array_lengthof(Aliases[i].Aliases); j++) {
      if (!Aliases[i].Aliases[j])
        break;
      if (Aliases[i].Aliases[j] == Name)
        return Aliases[i].Register;
    }
  }

  return Name;
}

bool TargetInfo::validateOutputConstraint(ConstraintInfo &Info) const {
  const char *Name = Info.getConstraintStr().c_str();
  // An output constraint must start with '=' or '+'
  if (*Name != '=' && *Name != '+')
    return false;

  if (*Name == '+')
    Info.setIsReadWrite();

  Name++;
  while (*Name) {
    switch (*Name) {
    default:
      if (!validateAsmConstraint(Name, Info)) {
        // FIXME: We temporarily return false
        // so we can add more constraints as we hit it.
        // Eventually, an unknown constraint should just be treated as 'g'.
        return false;
      }
    case '&': // early clobber.
      break;
    case '%': // commutative.
      // FIXME: Check that there is a another register after this one.
      break;
    case 'r': // general register.
      Info.setAllowsRegister();
      break;
    case 'm': // memory operand.
    case 'o': // offsetable memory operand.
    case 'V': // non-offsetable memory operand.
    case '<': // autodecrement memory operand.
    case '>': // autoincrement memory operand.
      Info.setAllowsMemory();
      break;
    case 'g': // general register, memory operand or immediate integer.
    case 'X': // any operand.
      Info.setAllowsRegister();
      Info.setAllowsMemory();
      break;
    case ',': // multiple alternative constraint.  Pass it.
      // Handle additional optional '=' or '+' modifiers.
      if (Name[1] == '=' || Name[1] == '+')
        Name++;
      break;
    case '?': // Disparage slightly code.
    case '!': // Disparage severely.
    case '#': // Ignore as constraint.
    case '*': // Ignore for choosing register preferences.
      break;  // Pass them.
    }

    Name++;
  }

  return true;
}

bool TargetInfo::resolveSymbolicName(const char *&Name,
                                     ConstraintInfo *OutputConstraints,
                                     unsigned NumOutputs,
                                     unsigned &Index) const {
  assert(*Name == '[' && "Symbolic name did not start with '['");
  Name++;
  const char *Start = Name;
  while (*Name && *Name != ']')
    Name++;

  if (!*Name) {
    // Missing ']'
    return false;
  }

  std::string SymbolicName(Start, Name - Start);

  for (Index = 0; Index != NumOutputs; ++Index)
    if (SymbolicName == OutputConstraints[Index].getName())
      return true;

  return false;
}

bool TargetInfo::validateInputConstraint(ConstraintInfo *OutputConstraints,
                                         unsigned NumOutputs,
                                         ConstraintInfo &Info) const {
  const char *Name = Info.ConstraintStr.c_str();

  while (*Name) {
    switch (*Name) {
    default:
      // Check if we have a matching constraint
      if (*Name >= '0' && *Name <= '9') {
        unsigned i = *Name - '0';

        // Check if matching constraint is out of bounds.
        if (i >= NumOutputs)
          return false;

        // A number must refer to an output only operand.
        if (OutputConstraints[i].isReadWrite())
          return false;

        // If the constraint is already tied, it must be tied to the 
        // same operand referenced to by the number.
        if (Info.hasTiedOperand() && Info.getTiedOperand() != i)
          return false;

        // The constraint should have the same info as the respective
        // output constraint.
        Info.setTiedOperand(i, OutputConstraints[i]);
      } else if (!validateAsmConstraint(Name, Info)) {
        // FIXME: This error return is in place temporarily so we can
        // add more constraints as we hit it.  Eventually, an unknown
        // constraint should just be treated as 'g'.
        return false;
      }
      break;
    case '[': {
      unsigned Index = 0;
      if (!resolveSymbolicName(Name, OutputConstraints, NumOutputs, Index))
        return false;

      // If the constraint is already tied, it must be tied to the 
      // same operand referenced to by the number.
      if (Info.hasTiedOperand() && Info.getTiedOperand() != Index)
        return false;

      Info.setTiedOperand(Index, OutputConstraints[Index]);
      break;
    }
    case '%': // commutative
      // FIXME: Fail if % is used with the last operand.
      break;
    case 'i': // immediate integer.
    case 'n': // immediate integer with a known value.
      break;
    case 'I':  // Various constant constraints with target-specific meanings.
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'P':
      break;
    case 'r': // general register.
      Info.setAllowsRegister();
      break;
    case 'm': // memory operand.
    case 'o': // offsettable memory operand.
    case 'V': // non-offsettable memory operand.
    case '<': // autodecrement memory operand.
    case '>': // autoincrement memory operand.
      Info.setAllowsMemory();
      break;
    case 'g': // general register, memory operand or immediate integer.
    case 'X': // any operand.
      Info.setAllowsRegister();
      Info.setAllowsMemory();
      break;
    case 'E': // immediate floating point.
    case 'F': // immediate floating point.
    case 'p': // address operand.
      break;
    case ',': // multiple alternative constraint.  Ignore comma.
      break;
    case '?': // Disparage slightly code.
    case '!': // Disparage severely.
    case '#': // Ignore as constraint.
    case '*': // Ignore for choosing register preferences.
      break;  // Pass them.
    }

    Name++;
  }

  return true;
}
