//===- CallDescription.cpp - function/method call matching     --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file defines a generic mechanism for matching for function and
/// method calls of C, C++, and Objective-C languages. Instances of these
/// classes are frequently used together with the CallEvent classes.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/Builtins.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/ADT/StringExtras.h"
#include <iterator>
#include <optional>

using namespace llvm;
using namespace clang;

using MaybeCount = std::optional<unsigned>;

// A constructor helper.
static MaybeCount readRequiredParams(MaybeCount RequiredArgs,
                                     MaybeCount RequiredParams) {
  if (RequiredParams)
    return RequiredParams;
  if (RequiredArgs)
    return RequiredArgs;
  return std::nullopt;
}

ento::CallDescription::CallDescription(Mode MatchAs,
                                       ArrayRef<StringRef> QualifiedName,
                                       MaybeCount RequiredArgs /*= None*/,
                                       MaybeCount RequiredParams /*= None*/)
    : RequiredArgs(RequiredArgs),
      RequiredParams(readRequiredParams(RequiredArgs, RequiredParams)),
      MatchAs(MatchAs) {
  assert(!QualifiedName.empty());
  this->QualifiedName.reserve(QualifiedName.size());
  llvm::transform(QualifiedName, std::back_inserter(this->QualifiedName),
                  [](StringRef From) { return From.str(); });
}

ento::CallDescription::Callee::Callee(const CallEvent &Call) {
  // FIXME: Add ObjC Message support.
  if (Call.getKind() == CE_ObjCMessage)
    return;

  FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD)
    return;

  ArgCount = Call.getNumArgs();
  ParamCount = FD->param_size();
  IsMethod = isa<CXXMethodDecl>(FD);
}

ento::CallDescription::Callee::Callee(const CallExpr &CE) {
  FD = dyn_cast_or_null<FunctionDecl>(CE.getCalleeDecl());
  if (!FD)
    return;

  ArgCount = CE.getNumArgs();
  ParamCount = FD->param_size();
  IsMethod = isa<CXXMethodDecl>(FD);
}

unsigned ento::CLibraryFunctionInfo::getBuiltinID() {
  if (!BuiltinID)
    BuiltinID = FD->getBuiltinID();
  return *BuiltinID;
}

StringRef ento::CLibraryFunctionInfo::getBuiltinName() {
  assert(getBuiltinID() != 0 && "Not a builtin");
  if (!BuiltinName)
    BuiltinName = FD->getASTContext().BuiltinInfo.getName(getBuiltinID());
  return *BuiltinName;
}

bool ento::CLibraryFunctionInfo::isPredefinedLibFunction() {
  assert(getBuiltinID() != 0 && "Not a builtin");
  if (!IsPredefinedLibFunction)
    IsPredefinedLibFunction =
        FD->getASTContext().BuiltinInfo.isPredefinedLibFunction(getBuiltinID());
  return *IsPredefinedLibFunction;
}

bool ento::CLibraryFunctionInfo::isCLibraryDecl() {
  if (IsCLibraryDecl)
    return *IsCLibraryDecl;

  IsCLibraryDecl = false;
  // If this is a special C++ name without IdentifierInfo, it can't be a
  // C library function.
  if (!FD->getIdentifier())
    return false;

  // C library functions are either declared directly within a TU (the common
  // case) or they are accessed through the namespace `std` (when they are used
  // in C++ via headers like <cstdlib>).
  const DeclContext *DC = FD->getDeclContext()->getRedeclContext();
  if (!(DC->isTranslationUnit() || DC->isStdNamespace()))
    return false;

  // If this function is not externally visible, it is not a C library function.
  // Note that we make an exception for inline functions, which may be
  // declared in header files without external linkage.
  IsCLibraryDecl = FD->isInlined() || FD->isExternallyVisible();
  return *IsCLibraryDecl;
}

bool ento::CLibraryFunctionInfo::isCLibraryFunction(StringRef Name) {
  // To avoid false positives (Ex: finding user defined functions with
  // similar names), only perform fuzzy name matching when it's a builtin.
  // Using a string compare is slow, we might want to switch on BuiltinID here.
  if (getBuiltinID() != 0) {
    if (Name.empty())
      return true;
    StringRef BName = getBuiltinName();
    // Accept exact match.
    if (BName == Name)
      return true;
    // Only perform fuzzy name matching for non-library builtins (e.g.
    // __builtin_memcpy, __builtin_memcpy_inline), not for distinct predefined
    // library functions (e.g. objc_read_weak vs read, getlogin_r vs getlogin).
    if (!isPredefinedLibFunction()) {
      size_t start = BName.find(Name);
      if (start != StringRef::npos) {
        //    v-- match starts here
        // ...xxxxx...
        //   _xxxxx_
        //   ^     ^ lookbehind and lookahead characters

        const auto MatchPredecessor = [&]() -> bool {
          return start <= 0 || !llvm::isAlpha(BName[start - 1]);
        };
        const auto MatchSuccessor = [&]() -> bool {
          std::size_t LookbehindPlace = start + Name.size();
          return LookbehindPlace >= BName.size() ||
                 !llvm::isAlpha(BName[LookbehindPlace]);
        };

        if (MatchPredecessor() && MatchSuccessor())
          return true;
      }
    }
  }

  if (!isCLibraryDecl())
    return false;

  if (Name.empty())
    return true;

  StringRef FName = FD->getName();
  if (FName == Name)
    return true;

  if (FName.starts_with("__inline") && FName.contains(Name))
    return true;

  return false;
}

bool ento::CallDescription::matches(const CallEvent &Call) const {
  Callee C(Call);
  CLibraryFunctionInfo CLib(C.FD);
  return matchesImpl(C, CLib);
}

bool ento::CallDescription::matchesAsWritten(const CallExpr &CE) const {
  Callee C(CE);
  CLibraryFunctionInfo CLib(C.FD);
  return matchesImpl(C, CLib);
}

bool ento::CallDescription::matchNameOnly(const NamedDecl *ND) const {
  DeclarationName Name = ND->getDeclName();
  if (const auto *NameII = Name.getAsIdentifierInfo()) {
    if (!II)
      II = &ND->getASTContext().Idents.get(getFunctionName());

    return NameII == *II; // Fast case.
  }

  // Avoid the stringification below where the printed name is known.
  StringRef FuncName = getFunctionName();
  switch (Name.getNameKind()) {
  case DeclarationName::CXXDestructorName:
    if (!FuncName.consume_front("~"))
      return false;
    [[fallthrough]];
  case DeclarationName::CXXConstructorName:
    // Constructor and destructor names are printed using the record name.
    if (const auto *RT = Name.getCXXNameType()->getAsCanonical<RecordType>()) {
      const RecordDecl *RD = RT->getDecl();
      if (const IdentifierInfo *RecordII = RD->getIdentifier())
        return RecordII->getName() == FuncName;
      // An unnamed record with a typedef name for linkage purposes is printed
      // using the typedef name.
      if (const TypedefNameDecl *TD = RD->getTypedefNameForAnonDecl())
        return TD->getIdentifier()->getName() == FuncName;
      // Other unnamed records are printed with their source location, e.g.
      // "(lambda at file:line:col)", which no description names.
      return false;
    }
    break;
  case DeclarationName::CXXOperatorName:
  case DeclarationName::CXXLiteralOperatorName:
  case DeclarationName::CXXConversionFunctionName:
    if (!FuncName.starts_with("operator"))
      return false;
    break;
  default:
    break;
  }

  // Fall back to printing the name, e.g. for overloaded, literal and conversion
  // operators.
  return Name.getAsString() == getFunctionName();
}

bool ento::CallDescription::matchQualifiedNameParts(const Decl *D) const {
  const auto FindNextNamespaceOrRecord =
      [](const DeclContext *Ctx) -> const DeclContext * {
    while (Ctx && !isa<NamespaceDecl, RecordDecl>(Ctx))
      Ctx = Ctx->getParent();
    return Ctx;
  };

  auto QualifierPartsIt = begin_qualified_name_parts();
  const auto QualifierPartsEndIt = end_qualified_name_parts();

  // Match namespace and record names. Skip unrelated names if they don't
  // match.
  const DeclContext *Ctx = FindNextNamespaceOrRecord(D->getDeclContext());
  for (; Ctx && QualifierPartsIt != QualifierPartsEndIt;
       Ctx = FindNextNamespaceOrRecord(Ctx->getParent())) {
    // If not matched just continue and try matching for the next one.
    if (cast<NamedDecl>(Ctx)->getName() != *QualifierPartsIt)
      continue;
    ++QualifierPartsIt;
  }

  // We matched if we consumed all expected qualifier segments.
  return QualifierPartsIt == QualifierPartsEndIt;
}

bool ento::CallDescription::matchesImpl(const Callee &C,
                                        CLibraryFunctionInfo &CLib) const {
  const FunctionDecl *FD = C.FD;
  if (!FD)
    return false;

  if (MatchAs == Mode::SimpleFunc && C.IsMethod)
    return false;

  if (MatchAs == Mode::CXXMethod && !C.IsMethod)
    return false;

  const size_t ArgCount = C.ArgCount;
  const size_t ParamCount = C.ParamCount;

  if (MatchAs == Mode::CLibraryMaybeHardened) {
    // In addition to accepting FOO() with CLibrary rules, we also want to
    // accept calls to __FOO_chk() and __builtin___FOO_chk().
    if (CheckerContext::isHardenedVariantOf(FD, getFunctionName()) &&
        CLib.isCLibraryFunction(StringRef())) {
      // Check that the actual argument/parameter counts are greater or equal
      // to the required counts. (Setting a requirement to std::nullopt matches
      // anything, so in that case value_or ensures that the value is compared
      // with itself.)
      return (RequiredArgs.value_or(ArgCount) <= ArgCount &&
              RequiredParams.value_or(ParamCount) <= ParamCount);
    }
  }

  if (RequiredArgs.value_or(ArgCount) != ArgCount ||
      RequiredParams.value_or(ParamCount) != ParamCount)
    return false;

  if (MatchAs == Mode::CLibrary || MatchAs == Mode::CLibraryMaybeHardened)
    return CLib.isCLibraryFunction(getFunctionName());

  if (!matchNameOnly(FD))
    return false;

  if (!hasQualifiedNameParts())
    return true;

  return matchQualifiedNameParts(FD);
}

ento::CallDescriptionSet::CallDescriptionSet(
    std::initializer_list<CallDescription> &&List) {
  Impl.LinearMap.reserve(List.size());
  for (const CallDescription &CD : List)
    Impl.LinearMap.push_back({CD, /*unused*/ true});
}

bool ento::CallDescriptionSet::contains(const CallEvent &Call) const {
  return static_cast<bool>(Impl.lookup(Call));
}

bool ento::CallDescriptionSet::containsAsWritten(const CallExpr &CE) const {
  return static_cast<bool>(Impl.lookupAsWritten(CE));
}
