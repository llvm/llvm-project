//===-- APINotesTypes.cpp - API Notes Data Types ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/APINotes/Types.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace api_notes {

// Detect whether top-level value const stripping would reach past the
// parameter itself.
// "const int" -> "int"
// "const int *" -> "const int*"
static bool
hasTopLevelIndirectParameterSelectorSpelling(llvm::StringRef Spelling) {
  unsigned Depth = 0;

  for (char C : Spelling) {
    switch (C) {
    case '*':
    case '&':
      if (Depth == 0)
        return true;
      break;

    case '[':
    case '(':
      if (Depth == 0)
        return true;
      ++Depth;
      break;

    case '<':
      ++Depth;
      break;

    case '>':
    case ']':
    case ')':
      if (Depth != 0)
        --Depth;
      break;

    default:
      break;
    }
  }

  return false;
}

// Return true when a space sits next to selector punctuation.
// The space in "int *" is dropped, but the space in "unsigned int" is kept.
static bool shouldDropParameterSelectorSpace(char Previous, char Next) {
  if (Previous == '<' || Previous == ',' || Next == '>' || Next == ',' ||
      Next == '<')
    return true;

  if (Next == '*' || Next == '&')
    return true;

  if (Previous == '&' && Next == '&')
    return true;

  return false;
}

// Collapse runs of whitespace between tokens.
// "  unsigned   int  " -> "unsigned int"
static std::string
collapseParameterSelectorWhitespace(llvm::StringRef Spelling) {
  llvm::SmallVector<llvm::StringRef, 4> Tokens;
  llvm::SplitString(Spelling, Tokens);
  return llvm::join(Tokens, " ");
}

// Normalize the short spelling for unsigned int.
// "unsigned" -> "unsigned int"
static llvm::StringRef normalizeUnsignedIntSpelling(llvm::StringRef Spelling) {
  if (Spelling == "unsigned")
    return "unsigned int";
  return Spelling;
}

// Strip const from by-value parameters only.
// "const int" -> "int", but "const int *" stays unchanged here.
static llvm::StringRef stripTopLevelValueConst(llvm::StringRef Spelling) {
  if (hasTopLevelIndirectParameterSelectorSpelling(Spelling))
    return Spelling;

  Spelling.consume_front("const ");
  Spelling.consume_back(" const");
  return Spelling;
}

// Remove spaces around selector punctuation while preserving token-separating
// spaces such as the one in "unsigned int".
// "int *" -> "int*", "Box<int, double>" -> "Box<int,double>"
static void removeParameterSelectorPunctuationSpaces(
    llvm::StringRef Spelling, llvm::SmallVectorImpl<char> &Normalized) {
  Normalized.clear();
  for (unsigned I = 0, E = Spelling.size(); I != E; ++I) {
    char C = Spelling[I];
    if (C == ' ' && I != 0 && I + 1 != E &&
        shouldDropParameterSelectorSpace(Spelling[I - 1], Spelling[I + 1]))
      continue;

    Normalized.push_back(C);
  }
}

// Strip const from the pointer object itself.
// "int* const" -> "int*", but "const int*" stays unchanged.
static std::string stripTopLevelPointerConst(llvm::StringRef Spelling) {
  if (!Spelling.consume_back("*const") && !Spelling.consume_back("* const"))
    return Spelling.str();

  std::string WithoutTopLevelConst = Spelling.str();
  WithoutTopLevelConst += '*';
  return WithoutTopLevelConst;
}

// Apply the full lexical selector normalization pipeline.
// " const  int " -> "int", " int  *  const " -> "int*"
std::string normalizeAPINotesParameterSelector(llvm::StringRef Spelling) {
  std::string Collapsed = collapseParameterSelectorWhitespace(Spelling);

  llvm::StringRef WithoutTopLevelValueConst =
      normalizeUnsignedIntSpelling(stripTopLevelValueConst(Collapsed));

  llvm::SmallString<32> WithoutPunctuationSpaces;
  removeParameterSelectorPunctuationSpaces(WithoutTopLevelValueConst,
                                           WithoutPunctuationSpaces);
  return stripTopLevelPointerConst(WithoutPunctuationSpaces);
}

LLVM_DUMP_METHOD void CommonEntityInfo::dump(llvm::raw_ostream &OS) const {
  if (Unavailable)
    OS << "[Unavailable] (" << UnavailableMsg << ")" << ' ';
  if (UnavailableInSwift)
    OS << "[UnavailableInSwift] ";
  if (SwiftPrivateSpecified)
    OS << (SwiftPrivate ? "[SwiftPrivate] " : "");
  if (SwiftSafetyAudited) {
    switch (*getSwiftSafety()) {
    case SwiftSafetyKind::Safe:
      OS << "[Safe] ";
      break;
    case SwiftSafetyKind::Unsafe:
      OS << "[Unsafe] ";
      break;
    case SwiftSafetyKind::Unspecified:
      OS << "[Unspecified] ";
      break;
    case SwiftSafetyKind::None:
      break;
    }
  }
  if (!SwiftName.empty())
    OS << "Swift Name: " << SwiftName << ' ';
  OS << '\n';
}

LLVM_DUMP_METHOD void CommonTypeInfo::dump(llvm::raw_ostream &OS) const {
  static_cast<const CommonEntityInfo &>(*this).dump(OS);
  if (SwiftBridge)
    OS << "Swift Briged Type: " << *SwiftBridge << ' ';
  if (NSErrorDomain)
    OS << "NSError Domain: " << *NSErrorDomain << ' ';
  OS << '\n';
}

LLVM_DUMP_METHOD void ContextInfo::dump(llvm::raw_ostream &OS) {
  static_cast<CommonTypeInfo &>(*this).dump(OS);
  if (NullabilityKindOrNone K = getDefaultNullability())
    OS << "DefaultNullability: " << *K << ' ';
  if (HasDesignatedInits)
    OS << "[HasDesignatedInits] ";
  if (SwiftImportAsNonGenericSpecified)
    OS << (SwiftImportAsNonGeneric ? "[SwiftImportAsNonGeneric] " : "");
  if (SwiftObjCMembersSpecified)
    OS << (SwiftObjCMembers ? "[SwiftObjCMembers] " : "");
  OS << '\n';
}

LLVM_DUMP_METHOD void VariableInfo::dump(llvm::raw_ostream &OS) const {
  static_cast<const CommonEntityInfo &>(*this).dump(OS);
  if (NullabilityKindOrNone K = getNullability())
    OS << "Audited Nullability: " << *K << ' ';
  if (!Type.empty())
    OS << "C Type: " << Type << ' ';
  OS << '\n';
}

LLVM_DUMP_METHOD void ObjCPropertyInfo::dump(llvm::raw_ostream &OS) const {
  static_cast<const VariableInfo &>(*this).dump(OS);
  if (SwiftImportAsAccessorsSpecified)
    OS << (SwiftImportAsAccessors ? "[SwiftImportAsAccessors] " : "");
  OS << '\n';
}

LLVM_DUMP_METHOD void BoundsSafetyInfo::dump(llvm::raw_ostream &OS) const {
  if (KindAudited) {
    switch (static_cast<BoundsSafetyKind>(Kind)) {
    case BoundsSafetyKind::CountedBy:
      OS << "[counted_by] ";
      break;
    case BoundsSafetyKind::CountedByOrNull:
      OS << "[counted_by_or_null] ";
      break;
    case BoundsSafetyKind::SizedBy:
      OS << "[sized_by] ";
      break;
    case BoundsSafetyKind::SizedByOrNull:
      OS << "[sized_by_or_null] ";
      break;
    case BoundsSafetyKind::EndedBy:
      OS << "[ended_by] ";
      break;
    }
  }
  if (LevelAudited)
    OS << "Level: " << Level << " ";
  OS << "ExternalBounds: "
     << (ExternalBounds.empty() ? "<missing>" : ExternalBounds) << '\n';
}

LLVM_DUMP_METHOD void ParamInfo::dump(llvm::raw_ostream &OS) const {
  static_cast<const VariableInfo &>(*this).dump(OS);
  if (NoEscapeSpecified)
    OS << (NoEscape ? "[NoEscape] " : "");
  if (LifetimeboundSpecified)
    OS << (Lifetimebound ? "[Lifetimebound] " : "");
  OS << "RawRetainCountConvention: " << RawRetainCountConvention << ' ';
  OS << '\n';
  if (BoundsSafety)
    BoundsSafety->dump(OS);
}

LLVM_DUMP_METHOD void FunctionInfo::dump(llvm::raw_ostream &OS) const {
  static_cast<const CommonEntityInfo &>(*this).dump(OS);
  OS << (NullabilityAudited ? "[NullabilityAudited] " : "")
     << (UnsafeBufferUsage ? "[UnsafeBufferUsage] " : "")
     << "RawRetainCountConvention: " << RawRetainCountConvention << ' ';
  if (!ResultType.empty())
    OS << "Result Type: " << ResultType << ' ';
  if (!SwiftReturnOwnership.empty())
    OS << "SwiftReturnOwnership: " << SwiftReturnOwnership << ' ';
  if (!Params.empty())
    OS << '\n';
  for (auto &PI : Params)
    PI.dump(OS);
}

LLVM_DUMP_METHOD void ObjCMethodInfo::dump(llvm::raw_ostream &OS) {
  static_cast<FunctionInfo &>(*this).dump(OS);
  if (Self)
    Self->dump(OS);
  OS << (DesignatedInit ? "[DesignatedInit] " : "")
     << (RequiredInit ? "[RequiredInit] " : "") << '\n';
}

LLVM_DUMP_METHOD void CXXMethodInfo::dump(llvm::raw_ostream &OS) {
  static_cast<FunctionInfo &>(*this).dump(OS);
  if (This)
    This->dump(OS);
}

LLVM_DUMP_METHOD void TagInfo::dump(llvm::raw_ostream &OS) {
  static_cast<CommonTypeInfo &>(*this).dump(OS);
  if (HasFlagEnum)
    OS << (IsFlagEnum ? "[FlagEnum] " : "");
  if (EnumExtensibility)
    OS << "Enum Extensibility: " << static_cast<long>(*EnumExtensibility)
       << ' ';
  if (SwiftCopyableSpecified)
    OS << (SwiftCopyable ? "[SwiftCopyable] " : "[~SwiftCopyable]");
  if (SwiftEscapableSpecified)
    OS << (SwiftEscapable ? "[SwiftEscapable] " : "[~SwiftEscapable]");
  OS << '\n';
}

LLVM_DUMP_METHOD void TypedefInfo::dump(llvm::raw_ostream &OS) const {
  static_cast<const CommonTypeInfo &>(*this).dump(OS);
  if (SwiftWrapper)
    OS << "Swift Type: " << static_cast<long>(*SwiftWrapper) << ' ';
  OS << '\n';
}
} // namespace api_notes
} // namespace clang
