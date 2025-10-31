//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeMismatchCheck.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"
#include "llvm/ADT/StringSet.h"
#include <map>

using namespace clang::ast_matchers;

namespace clang::tidy::mpi {

/// Check if a BuiltinType::Kind matches the MPI datatype.
///
/// \param MultiMap datatype group
/// \param Kind buffer type kind
/// \param MPIDatatype name of the MPI datatype
///
/// \returns true if the pair matches
static bool
isMPITypeMatching(const std::multimap<BuiltinType::Kind, StringRef> &MultiMap,
                  const BuiltinType::Kind Kind, StringRef MPIDatatype) {
  auto ItPair = MultiMap.equal_range(Kind);
  while (ItPair.first != ItPair.second) {
    if (ItPair.first->second == MPIDatatype)
      return true;
    ++ItPair.first;
  }
  return false;
}

/// Check if the MPI datatype is a standard type.
///
/// \param MPIDatatype name of the MPI datatype
///
/// \returns true if the type is a standard type
static bool isStandardMPIDatatype(StringRef MPIDatatype) {
  static llvm::StringSet<> AllTypes = {"MPI_C_BOOL",
                                       "MPI_CHAR",
                                       "MPI_SIGNED_CHAR",
                                       "MPI_UNSIGNED_CHAR",
                                       "MPI_WCHAR",
                                       "MPI_INT",
                                       "MPI_LONG",
                                       "MPI_SHORT",
                                       "MPI_LONG_LONG",
                                       "MPI_LONG_LONG_INT",
                                       "MPI_UNSIGNED",
                                       "MPI_UNSIGNED_SHORT",
                                       "MPI_UNSIGNED_LONG",
                                       "MPI_UNSIGNED_LONG_LONG",
                                       "MPI_FLOAT",
                                       "MPI_DOUBLE",
                                       "MPI_LONG_DOUBLE",
                                       "MPI_C_COMPLEX",
                                       "MPI_C_FLOAT_COMPLEX",
                                       "MPI_C_DOUBLE_COMPLEX",
                                       "MPI_C_LONG_DOUBLE_COMPLEX",
                                       "MPI_INT8_T",
                                       "MPI_INT16_T",
                                       "MPI_INT32_T",
                                       "MPI_INT64_T",
                                       "MPI_UINT8_T",
                                       "MPI_UINT16_T",
                                       "MPI_UINT32_T",
                                       "MPI_UINT64_T",
                                       "MPI_CXX_BOOL",
                                       "MPI_CXX_FLOAT_COMPLEX",
                                       "MPI_CXX_DOUBLE_COMPLEX",
                                       "MPI_CXX_LONG_DOUBLE_COMPLEX"};

  return AllTypes.contains(MPIDatatype);
}

/// Check if a BuiltinType matches the MPI datatype.
///
/// \param Builtin the builtin type
/// \param BufferTypeName buffer type name, gets assigned
/// \param MPIDatatype name of the MPI datatype
/// \param LO language options
///
/// \returns true if the type matches
static bool isBuiltinTypeMatching(const BuiltinType *Builtin,
                                  std::string &BufferTypeName,
                                  StringRef MPIDatatype,
                                  const LangOptions &LO) {
  static std::multimap<BuiltinType::Kind, StringRef> BuiltinMatches = {
      // On some systems like PPC or ARM, 'char' is unsigned by default which is
      // why distinct signedness for the buffer and MPI type is tolerated.
      {BuiltinType::SChar, "MPI_CHAR"},
      {BuiltinType::SChar, "MPI_SIGNED_CHAR"},
      {BuiltinType::SChar, "MPI_UNSIGNED_CHAR"},
      {BuiltinType::Char_S, "MPI_CHAR"},
      {BuiltinType::Char_S, "MPI_SIGNED_CHAR"},
      {BuiltinType::Char_S, "MPI_UNSIGNED_CHAR"},
      {BuiltinType::UChar, "MPI_CHAR"},
      {BuiltinType::UChar, "MPI_SIGNED_CHAR"},
      {BuiltinType::UChar, "MPI_UNSIGNED_CHAR"},
      {BuiltinType::Char_U, "MPI_CHAR"},
      {BuiltinType::Char_U, "MPI_SIGNED_CHAR"},
      {BuiltinType::Char_U, "MPI_UNSIGNED_CHAR"},
      {BuiltinType::WChar_S, "MPI_WCHAR"},
      {BuiltinType::WChar_U, "MPI_WCHAR"},
      {BuiltinType::Bool, "MPI_C_BOOL"},
      {BuiltinType::Bool, "MPI_CXX_BOOL"},
      {BuiltinType::Short, "MPI_SHORT"},
      {BuiltinType::Int, "MPI_INT"},
      {BuiltinType::Long, "MPI_LONG"},
      {BuiltinType::LongLong, "MPI_LONG_LONG"},
      {BuiltinType::LongLong, "MPI_LONG_LONG_INT"},
      {BuiltinType::UShort, "MPI_UNSIGNED_SHORT"},
      {BuiltinType::UInt, "MPI_UNSIGNED"},
      {BuiltinType::ULong, "MPI_UNSIGNED_LONG"},
      {BuiltinType::ULongLong, "MPI_UNSIGNED_LONG_LONG"},
      {BuiltinType::Float, "MPI_FLOAT"},
      {BuiltinType::Double, "MPI_DOUBLE"},
      {BuiltinType::LongDouble, "MPI_LONG_DOUBLE"}};

  if (!isMPITypeMatching(BuiltinMatches, Builtin->getKind(), MPIDatatype)) {
    BufferTypeName = std::string(Builtin->getName(LO));
    return false;
  }

  return true;
}

/// Check if a complex float/double/long double buffer type matches
/// the MPI datatype.
///
/// \param Complex buffer type
/// \param BufferTypeName buffer type name, gets assigned
/// \param MPIDatatype name of the MPI datatype
/// \param LO language options
///
/// \returns true if the type matches or the buffer type is unknown
static bool isCComplexTypeMatching(const ComplexType *const Complex,
                                   std::string &BufferTypeName,
                                   StringRef MPIDatatype,
                                   const LangOptions &LO) {
  static std::multimap<BuiltinType::Kind, StringRef> ComplexCMatches = {
      {BuiltinType::Float, "MPI_C_COMPLEX"},
      {BuiltinType::Float, "MPI_C_FLOAT_COMPLEX"},
      {BuiltinType::Double, "MPI_C_DOUBLE_COMPLEX"},
      {BuiltinType::LongDouble, "MPI_C_LONG_DOUBLE_COMPLEX"}};

  const auto *Builtin =
      Complex->getElementType().getTypePtr()->getAs<BuiltinType>();

  if (Builtin &&
      !isMPITypeMatching(ComplexCMatches, Builtin->getKind(), MPIDatatype)) {
    BufferTypeName = (llvm::Twine(Builtin->getName(LO)) + " _Complex").str();
    return false;
  }
  return true;
}

/// Check if a complex<float/double/long double> templated buffer type matches
/// the MPI datatype.
///
/// \param Template buffer type
/// \param BufferTypeName buffer type name, gets assigned
/// \param MPIDatatype name of the MPI datatype
/// \param LO language options
///
/// \returns true if the type matches or the buffer type is unknown
static bool
isCXXComplexTypeMatching(const TemplateSpecializationType *const Template,
                         std::string &BufferTypeName, StringRef MPIDatatype,
                         const LangOptions &LO) {
  static std::multimap<BuiltinType::Kind, StringRef> ComplexCXXMatches = {
      {BuiltinType::Float, "MPI_CXX_FLOAT_COMPLEX"},
      {BuiltinType::Double, "MPI_CXX_DOUBLE_COMPLEX"},
      {BuiltinType::LongDouble, "MPI_CXX_LONG_DOUBLE_COMPLEX"}};

  if (Template->getAsCXXRecordDecl()->getName() != "complex")
    return true;

  const auto *Builtin = Template->template_arguments()[0]
                            .getAsType()
                            .getTypePtr()
                            ->getAs<BuiltinType>();

  if (Builtin &&
      !isMPITypeMatching(ComplexCXXMatches, Builtin->getKind(), MPIDatatype)) {
    BufferTypeName =
        (llvm::Twine("complex<") + Builtin->getName(LO) + ">").str();
    return false;
  }

  return true;
}

/// Check if a fixed size width buffer type matches the MPI datatype.
///
/// \param Typedef buffer type
/// \param BufferTypeName buffer type name, gets assigned
/// \param MPIDatatype name of the MPI datatype
///
/// \returns true if the type matches or the buffer type is unknown
static bool isTypedefTypeMatching(const TypedefType *const Typedef,
                                  std::string &BufferTypeName,
                                  StringRef MPIDatatype) {
  static llvm::StringMap<StringRef> FixedWidthMatches = {
      {"int8_t", "MPI_INT8_T"},     {"int16_t", "MPI_INT16_T"},
      {"int32_t", "MPI_INT32_T"},   {"int64_t", "MPI_INT64_T"},
      {"uint8_t", "MPI_UINT8_T"},   {"uint16_t", "MPI_UINT16_T"},
      {"uint32_t", "MPI_UINT32_T"}, {"uint64_t", "MPI_UINT64_T"}};

  const auto It = FixedWidthMatches.find(Typedef->getDecl()->getName());
  // Check if the typedef is known and not matching the MPI datatype.
  if (It != FixedWidthMatches.end() && It->getValue() != MPIDatatype) {
    BufferTypeName = std::string(Typedef->getDecl()->getName());
    return false;
  }
  return true;
}

/// Get the unqualified, dereferenced type of an argument.
///
/// \param CE call expression
/// \param Idx argument index
///
/// \returns type of the argument
static const Type *argumentType(const CallExpr *const CE, const size_t Idx) {
  const QualType QT = CE->getArg(Idx)->IgnoreImpCasts()->getType();
  return QT.getTypePtr()->getPointeeOrArrayElementType();
}

void TypeMismatchCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(callExpr().bind("CE"), this);
}

void TypeMismatchCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *const CE = Result.Nodes.getNodeAs<CallExpr>("CE");
  if (!CE->getDirectCallee())
    return;

  if (!FuncClassifier)
    FuncClassifier.emplace(*Result.Context);

  const IdentifierInfo *Identifier = CE->getDirectCallee()->getIdentifier();
  if (!Identifier || !FuncClassifier->isMPIType(Identifier))
    return;

  // These containers are used, to capture buffer, MPI datatype pairs.
  SmallVector<const Type *, 1> BufferTypes;
  SmallVector<const Expr *, 1> BufferExprs;
  SmallVector<StringRef, 1> MPIDatatypes;

  // Adds a buffer, MPI datatype pair of an MPI call expression to the
  // containers. For buffers, the type and expression is captured.
  auto AddPair = [&CE, &Result, &BufferTypes, &BufferExprs, &MPIDatatypes](
                     const size_t BufferIdx, const size_t DatatypeIdx) {
    // Skip null pointer constants and in place 'operators'.
    if (CE->getArg(BufferIdx)->isNullPointerConstant(
            *Result.Context, Expr::NPC_ValueDependentIsNull) ||
        tooling::fixit::getText(*CE->getArg(BufferIdx), *Result.Context) ==
            "MPI_IN_PLACE")
      return;

    StringRef MPIDatatype =
        tooling::fixit::getText(*CE->getArg(DatatypeIdx), *Result.Context);

    const Type *ArgType = argumentType(CE, BufferIdx);
    // Skip unknown MPI datatypes and void pointers.
    if (!isStandardMPIDatatype(MPIDatatype) || ArgType->isVoidType())
      return;

    BufferTypes.push_back(ArgType);
    BufferExprs.push_back(CE->getArg(BufferIdx));
    MPIDatatypes.push_back(MPIDatatype);
  };

  // Collect all buffer, MPI datatype pairs for the inspected call expression.
  if (FuncClassifier->isPointToPointType(Identifier)) {
    AddPair(0, 2);
  } else if (FuncClassifier->isCollectiveType(Identifier)) {
    if (FuncClassifier->isReduceType(Identifier)) {
      AddPair(0, 3);
      AddPair(1, 3);
    } else if (FuncClassifier->isScatterType(Identifier) ||
               FuncClassifier->isGatherType(Identifier) ||
               FuncClassifier->isAlltoallType(Identifier)) {
      AddPair(0, 2);
      AddPair(3, 5);
    } else if (FuncClassifier->isBcastType(Identifier)) {
      AddPair(0, 2);
    }
  }
  checkArguments(BufferTypes, BufferExprs, MPIDatatypes, getLangOpts());
}

void TypeMismatchCheck::checkArguments(ArrayRef<const Type *> BufferTypes,
                                       ArrayRef<const Expr *> BufferExprs,
                                       ArrayRef<StringRef> MPIDatatypes,
                                       const LangOptions &LO) {
  std::string BufferTypeName;

  for (size_t I = 0; I < MPIDatatypes.size(); ++I) {
    const Type *const BT = BufferTypes[I];
    bool Error = false;

    if (const auto *Typedef = BT->getAs<TypedefType>()) {
      Error = !isTypedefTypeMatching(Typedef, BufferTypeName, MPIDatatypes[I]);
    } else if (const auto *Complex = BT->getAs<ComplexType>()) {
      Error =
          !isCComplexTypeMatching(Complex, BufferTypeName, MPIDatatypes[I], LO);
    } else if (const auto *Template = BT->getAs<TemplateSpecializationType>()) {
      Error = !isCXXComplexTypeMatching(Template, BufferTypeName,
                                        MPIDatatypes[I], LO);
    } else if (const auto *Builtin = BT->getAs<BuiltinType>()) {
      Error =
          !isBuiltinTypeMatching(Builtin, BufferTypeName, MPIDatatypes[I], LO);
    }

    if (Error) {
      const auto Loc = BufferExprs[I]->getSourceRange().getBegin();
      diag(Loc, "buffer type '%0' does not match the MPI datatype '%1'")
          << BufferTypeName << MPIDatatypes[I];
    }
  }
}

void TypeMismatchCheck::onEndOfTranslationUnit() { FuncClassifier.reset(); }
} // namespace clang::tidy::mpi
