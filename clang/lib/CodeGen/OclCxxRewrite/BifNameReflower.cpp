//===- BifNameReflower.cpp - Built-in name reflower pass(OCLC++)-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//
//
// Copyright (c) 2015 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
//
//===----------------------------------------------------------------------===//


#include "clang/CodeGen/OclCxxRewrite/BifNameReflower.h"

#include "OclCxxDemangler.h"
#include "OclCxxPrinter.h"

#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"

#include <tuple>
#include <utility>

#define DEBUG_TYPE "OclCxxBifNameReflower"


namespace {
using namespace oclcxx::adaptation;


/// \brief Node adaptation traits for Itanium-mangling printer which
///        adapt names and types from OpenCL C++ to interface layer of
///        SPIR-V generator.
struct OclCxxBiAdaptTraits {
  /// \brief Indicates that parsed name is candidate for built-in name
  ///        reflow.
  ///
  /// \param ParsedName              Result returned from demangler parsing.
  ///                                Result can be failed.
  /// \param AllowRttiTypeNameString Indicates that RTTI type strings with
  ///                                type names are allowed to be reflown.
  /// \return                        true if current name is valid candidate
  ///                                for reflow; otherwise, false.
  static bool isCandidateForReflow(const DmngRslt &ParsedName,
                                   bool AllowRttiTypeNameString = false) {
      // Only process ordinary function names which have mangled name in
      // cl::__spirv namespace.
      if (!ParsedName.isSuccessful())
        return false;
      auto PName = ParsedName.getName();

      // If RTTI type string is allowed, go to related type name and use it
      // as base name for candidate analysis.
      if (AllowRttiTypeNameString && PName->isSpecial()) {
        auto PSpecName = PName->getAs<DNK_Special>();
        if (PSpecName->getSpecialKind() != DSNK_TypeInfoNameString)
          return false;
        auto PRelTypeName = PSpecName->getRelatedType()->getAs<DTK_TypeName>();
        if (PRelTypeName == nullptr)
          return false;
        PName = PRelTypeName->getTypeName();
      }

      if (!PName->isOrdinary())
        return false;
      auto POrdName = PName->getAs<DNK_Ordinary>();
      if (POrdName->isLocal())
        return false;
      const auto &PNameParts = POrdName->getParts();
      if (PNameParts.size() <= 2)
        return false;
      auto PFirstNamePart = PNameParts[0]->getAs<DNPK_Source>();
      auto PSecondNamePart = PNameParts[1]->getAs<DNPK_Source>();
      if (PFirstNamePart == nullptr || PSecondNamePart == nullptr ||
          PFirstNamePart->isTemplate() || PSecondNamePart->isTemplate() ||
          PFirstNamePart->getSourceName() != "cl" ||
          PSecondNamePart->getSourceName() != "__spirv")
        return false;

      return true;
  }

private:
  /// \brief Built-in entity category.
  enum class BiCategory {
    Failure,    ///< Category is unknown. Failed to detect category.
    Function,   ///< Built-in function (either in extended set or core "Op"
                ///< function).
    Type,       ///< Built-in type (core "OpType"-function-generated).
    Variable,   ///< Built-in variable (core "Var"-prefixed global variable).
  };

  /// \brief Gets extended set and built-in names based on candidate name.
  ///
  /// \param NameNode   Node in result that describes name of built-in entity.
  /// \return           Triple containing name of extended set (first argument),
  ///                   name of built-in entity (second argument) and its
  ///                   category (third argument).
  ///
  ///                   If method failed, the triple with empty strings
  ///                   and BiCategory::Failure is returned.
  static std::tuple<std::string, std::string, BiCategory> getExtSetBiName(
      const std::shared_ptr<const DmngRsltOrdinaryName> &NameNode) {
    assert(NameNode != nullptr && "Node must exist.");

    // If function is directly in a subnamespace of cl::__spirv namespace and
    // it does not name any special function then it will be treated as
    // built-in from extended set named the same as subnamespace name.
    //
    // If function is directly in cl::__spirv namespace and it does not name
    // any special function then it will be treated as built-in name.
    // If the built-in name starts with "Op" prefix, it is core built-in;
    // otherwise, it is built-in from "ocl" extended set.
    const auto &PNameParts = NameNode->getParts();

    std::string ExtSetName;
    std::string BiName;
    BiCategory Category = BiCategory::Failure;
    switch (PNameParts.size()) {
    case 4: {
        auto PExtSetNamePart = PNameParts[2]->getAs<DNPK_Source>();
        if (PExtSetNamePart == nullptr || PExtSetNamePart->isTemplate())
          return std::make_tuple(std::string(), std::string(), Category);
        ExtSetName = PExtSetNamePart->getSourceName();
      }
      break;
    case 3: break; // ExtSetName = "";
    default:
      return std::make_tuple(std::string(), std::string(), Category);
    }

    auto PBiNamePart = PNameParts.back()->getAs<DNPK_Source>();
    if (PBiNamePart == nullptr)
      return std::make_tuple(std::string(), std::string(), Category);
    if (ExtSetName.empty() &&
        PBiNamePart->getSourceName().compare(0, 2, "Op") == 0)
      if (NameNode->isData() &&
          PBiNamePart->getSourceName().compare(2, 4, "Type") == 0) {
        BiName = PBiNamePart->getSourceName().substr(6);
        Category = BiCategory::Type;
      }
      else if (NameNode->isData() &&
          PBiNamePart->getSourceName().compare(2, 8, "Constant") == 0) {
        BiName = PBiNamePart->getSourceName().substr(2);
        Category = BiCategory::Type;
      }
      else {
        BiName = PBiNamePart->getSourceName().substr(2);
        Category = BiCategory::Function;
      }
    else if (ExtSetName.empty() && NameNode->isData() &&
             PBiNamePart->getSourceName().compare(0, 3, "Var") == 0) {
      BiName = PBiNamePart->getSourceName().substr(3);
      Category = BiCategory::Variable;
    }
    else {
      if (ExtSetName.empty())
        ExtSetName = "ocl";
      BiName = PBiNamePart->getSourceName();
      Category = BiCategory::Function;
    }

    return std::make_tuple(ExtSetName, BiName, Category);
  }


  /// \brief Gets effective number of elements in array or vector.
  ///
  /// \param TypeNode Node in demangler's result which points to array or
  ///                 vector type. The method asserts, if node is invalid.
  /// \return         Computed effective number of elements, or 0 if function
  ///                 cannot compute it (unspecified or variable).
  static unsigned long long getEffectiveArrayVecElements(
      const std::shared_ptr<const DmngRsltArrayVecType> &TypeNode) {
    assert(TypeNode != nullptr && "Node must exist.");

    // Simple size.
    if (TypeNode->getSizeExpr() == nullptr)
      return TypeNode->getSize();

    // Size is an expression. If it is template parameter, resolve it
    // to final expression if possible.
    auto SizeExpr = TypeNode->getSizeExpr();
    while (SizeExpr != nullptr && SizeExpr->getKind() == DXK_TemplateParam) {
      auto SizeTArg = SizeExpr->getAs<DXK_TemplateParam>()->getReferredTArg();
      if (SizeTArg == nullptr || !SizeTArg->isExpression())
        return 0;
      SizeExpr = SizeTArg->getExpression();
    }

    // If final size expression is primary expression (literal), get its
    // value.
    auto PrimExpr = SizeExpr->getAs<DXK_Primary>();
    if (PrimExpr != nullptr && PrimExpr->isLiteral()) {
      switch (PrimExpr->getContentType()) {
      case DmngRsltPrimaryExpr::UInt:
      case DmngRsltPrimaryExpr::Bool:
        return PrimExpr->getContentAsUInt();
      case DmngRsltPrimaryExpr::SInt:
        return PrimExpr->getContentAsSInt() > 0
                 ? static_cast<unsigned long long>(PrimExpr->getContentAsSInt())
                 : 0;
      default:
        return 0;
      }
    }

    // Fall, if size cannot be computed.
    return 0;
  }

  /// \brief Gets effective type.
  ///
  /// \param TypeNode Node in demangler's result which points to type.
  ///                 The method asserts, if node is invalid.
  /// \return         Effective type of TypeNode. The effective type is a type
  ///                 after resolving template parameter or decltype()
  ///                 expressions.
  ///                 If type cannot be resolved, empty shared pointer is
  ///                 returned.
  static std::shared_ptr<const DmngRsltType> getEffectiveType(
      const std::shared_ptr<const DmngRsltType> &TypeNode) {
    assert(TypeNode != nullptr && "Node must exist.");

    // If type node is template parameter or decltype() node,
    // resolve it to final type.
    auto FinalNode = TypeNode;
    while (FinalNode != nullptr && (FinalNode->getKind() == DTK_TemplateParam ||
                                    FinalNode->getKind() == DTK_Decltype)) {
      std::shared_ptr<const DmngRsltExpr> TypeExpr;
      if (FinalNode->getKind() == DTK_TemplateParam)
        TypeExpr = FinalNode->getAs<DTK_TemplateParam>()->getTemplateParam();
      else
        TypeExpr = FinalNode->getAs<DTK_Decltype>()->getDecltype();

      // If expression node is template parameter expression, go to referred
      // type or inner expression. If expression node is decltype() expression,
      // go to inner expression.
      FinalNode = nullptr;
      while (TypeExpr != nullptr && (TypeExpr->getKind() == DXK_TemplateParam ||
                                     TypeExpr->getKind() == DXK_Decltype)) {
        if (TypeExpr->getKind() == DXK_TemplateParam) {
          auto TArg = TypeExpr->getAs<DXK_TemplateParam>()->getReferredTArg();
          if (TArg != nullptr) {
            if (TArg->isExpression()) {
              TypeExpr = TArg->getExpression();
              continue;
            }
            if (TArg->isType())
              FinalNode = TArg->getType();
          }
          break;
        }
        TypeExpr = TypeExpr->getAs<DXK_Decltype>()->getExpression();
      }
    }

    return FinalNode;
  }

  /// \brief Gets effective type.
  ///
  /// \param TArg Template argument in demangler's result.
  /// \return     Effective type of TArg. The effective type is a type stored in
  ///             template arg after resolving template parameter or decltype()
  ///             expressions.
  ///             If type cannot be resolved, empty shared pointer is returned.
  static std::shared_ptr<const DmngRsltType> getEffectiveType(
      const DmngRsltTArg &TArg) {
    const auto *EffectiveTArg = &TArg;

    while (EffectiveTArg != nullptr) {
      if (EffectiveTArg->isType())
        return getEffectiveType(EffectiveTArg->getType());

      if (!EffectiveTArg->isExpression())
        return nullptr;

      auto Expr = EffectiveTArg->getExpression();
      EffectiveTArg = nullptr;
      while (Expr != nullptr) {
        switch (Expr->getKind()) {
        case DXK_TemplateParam:
          EffectiveTArg = Expr->getAs<DXK_TemplateParam>()
                            ->getReferredTArg().get();
          Expr = nullptr;
          break;
        case DXK_Decltype:
          Expr = Expr->getAs<DXK_Decltype>()->getExpression();
          break;
        default:
          return nullptr;
        }
      }
    }
    return nullptr;
  }

  /// \brief Gets effective primary expression (literal or external name)
  ///        stored in template argument.
  ///
  /// \param TArg Template argument in demangler's result.
  /// \return     Effective primary expression stored in template argument.
  ///             The effective expression is a expression after resolving
  ///             template parameter or decltype() expressions.
  ///             If expression cannot be resolved, empty shared pointer
  ///             is returned.
  static std::shared_ptr<const DmngRsltPrimaryExpr> getEffectivePrimaryExpr(
      const DmngRsltTArg &TArg) {
    const auto *EffectiveTArg = &TArg;

    while (EffectiveTArg != nullptr) {
      if (!EffectiveTArg->isExpression())
        return nullptr;
      auto Expr = EffectiveTArg->getExpression();
      switch (Expr->getKind()) {
      case DXK_TemplateParam:
        EffectiveTArg = Expr->getAs<DXK_TemplateParam>()
                          ->getReferredTArg().get();
        break;
      case DXK_Primary:
        return Expr->getAs<DXK_Primary>();
      default:
        return nullptr;
      }
    }
    return nullptr;
  }


  /// \brief Gets encoding for special return types.
  ///
  /// \param TypeNode           Node in demangler's result which points to
  ///                           (return) type.
  ///                           The method asserts, if node is invalid.
  /// \param AllowVoidEncoding  Encoding void type is only allowed in
  ///                           the special cases.
  /// \return                   Encoded name of special return type, or empty
  ///                           string if type is not special or type is
  ///                           invalid.
  static std::string getSpecialReturnTypeEncoding(
      const std::shared_ptr<const DmngRsltType> &TypeNode,
      bool AllowVoidEncoding = false) {
    auto EffectiveType = getEffectiveType(TypeNode);
    if (EffectiveType == nullptr)
      return std::string();

    // Vector type suffix.
    std::string VecSuffix;
    auto EffectiveVecType = EffectiveType->getAs<DTK_Vector>();
    if (EffectiveVecType != nullptr) {
      auto VecSize = getEffectiveArrayVecElements(EffectiveVecType);
      switch (VecSize) {
      case 1:  break;
      case 2:  VecSuffix = "2"; break;
      case 3:  VecSuffix = "3"; break;
      case 4:  VecSuffix = "4"; break;
      case 8:  VecSuffix = "8"; break;
      case 16: VecSuffix = "16"; break;
      default:
        return std::string(); // Invalid vector size.
      }

      EffectiveType = getEffectiveType(EffectiveVecType->getElemType());
      if (EffectiveType == nullptr)
        return std::string(); // Invalid element type.
    }

    // Scalar type or vector element type.
    std::string ScalarType;
    auto EffectBiType = EffectiveType->getAs<DTK_Builtin>();
    if (EffectBiType != nullptr) {
      switch (EffectBiType->getBuiltinType()) {
      case DBT_Void:
        if (AllowVoidEncoding && VecSuffix.empty())
          ScalarType = "void";
        else
          return std::string(); break;
      case DBT_Char:
      case DBT_SChar:    ScalarType = "char";   break;
      case DBT_UChar:    ScalarType = "uchar";  break;
      case DBT_Short:    ScalarType = "short";  break;
      case DBT_UShort:   ScalarType = "ushort"; break;
      case DBT_Int:      ScalarType = "int";    break;
      case DBT_UInt:     ScalarType = "uint";   break;
      case DBT_Long:
      case DBT_Int64:    ScalarType = "long";   break;
      case DBT_ULong:
      case DBT_UInt64:   ScalarType = "ulong";  break;
      case DBT_Half:     ScalarType = "half";   break;
      case DBT_Float:
      case DBT_Float32R: ScalarType = "float";  break;
      case DBT_Double:
      case DBT_Float64R: ScalarType = "double"; break;
      case DBT_Bool:     ScalarType = "bool";   break;
      default:
        return std::string();
      }
    }

    return ScalarType + VecSuffix;
  }

  /// \brief Type of special enum to encode in name.
  enum class SpecialEnumType {
    Sat,         ///< Saturation option.
    RoundingMode ///< Rounding mode option.
  };

  /// \brief Get encoding for special enumeration values in template argument.
  ///
  /// Stops scanning on first encountered special enum value.
  ///
  /// \tparam EnumType Type of special enum to scan for.
  ///
  /// \param TArg      Template argument in demangler's result to scan for
  ///                  special enum values.
  /// \param ScanPacks Indicates that nested template parameter packs should
  ///                  also be scanned.
  /// \return          Encoded name of special enum value, or nullptr if
  ///                  template argument does not contain special value.
  template<SpecialEnumType EnumType>
  static const char *getSpecialEnumValueEncoding(const DmngRsltTArg &TArg,
                                                 bool ScanPacks = false) {
    if (ScanPacks && TArg.isPack())
      return getSpecialEnumValueEncoding<EnumType>(TArg.getPack(), ScanPacks);

    // Resolve enum type name.
    auto EffectiveExpr = getEffectivePrimaryExpr(TArg);
    if (EffectiveExpr == nullptr || !EffectiveExpr->isLiteral())
      return nullptr;
    auto EffectiveType = getEffectiveType(EffectiveExpr->getLiteralType());
    if (EffectiveType == nullptr || EffectiveType->getKind() != DTK_TypeName)
      return nullptr;
    auto LiteralTypeNameType = EffectiveType->getAs<DTK_TypeName>();
    if (LiteralTypeNameType->getElaboration() != DTNK_None &&
        LiteralTypeNameType->getElaboration() != DTNK_ElaboratedEnum)
      return nullptr;
    auto LiteralTypeName = LiteralTypeNameType->getTypeName();
    if (!LiteralTypeName->isData())
      return nullptr;

    // Check type name for proper parts.
    if (LiteralTypeName->getParts().size() != 2)
      return nullptr;
    auto NsPart = LiteralTypeName->getParts()[0]->getAs<DNPK_Source>();
    auto EnumPart = LiteralTypeName->getParts()[1]->getAs<DNPK_Source>();
    if (NsPart == nullptr || EnumPart == nullptr ||
        NsPart->getSourceName() != "cl")
      return nullptr;


    // enum class cl::saturate     { off, on };
    if (EnumType == SpecialEnumType::Sat) {
      if (EnumPart->getSourceName() != "saturate")
        return nullptr;

      switch (EffectiveExpr->getContentAsUInt()) {
      case 0: // off
        return "";
      case 1: // on
        return "_sat";
      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unsupported enum value for cl::saturate.");
        return nullptr;
      }
    }

    // enum class cl::rounding_mode { rte, rtz, rtp, rtn };
    if (EnumType == SpecialEnumType::RoundingMode) {
      if (EnumPart->getSourceName() != "rounding_mode")
        return nullptr;

      switch (EffectiveExpr->getContentAsUInt()) {
      case 0: // rte
        return "_rte";
      case 1: // rtz
        return "_rtz";
      case 2: // rtp
        return "_rtp";
      case 3: // rtn
        return "_rtn";
      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unsupported enum value for cl::rounding_mode.");
        return nullptr;
      }
    }

    return nullptr;
  }

  /// \brief Get encoding for special enumeration values in template arguments.
  ///
  /// Stops scanning on first encountered special enum value.
  ///
  /// \tparam EnumType Type of special enum to scan for.
  ///
  /// \param TArgs     Template arguments collection in demangler's result
  ///                  to scan for special enum values.
  /// \param ScanPacks Indicates that nested template parameter packs should
  ///                  also be scanned.
  /// \return          Encoded name of special enum value, or nullptr if
  ///                  template arguments do not contain special value.
  template<SpecialEnumType EnumType>
  static const char *getSpecialEnumValueEncoding(
      const DmngRsltTArg::TArgsPackColT &TArgs, bool ScanPacks = false) {
    for (const auto &TArg : TArgs) {
      auto EnumVal = getSpecialEnumValueEncoding<EnumType>(TArg, ScanPacks);
      if (EnumVal != nullptr)
        return EnumVal;
    }
    return nullptr;
  }


  /// \brief Encodes template argument of data type (or data variable).
  ///
  /// If parameter packs are also scanned, they are flattened and their template
  /// arguments are encoded in order they appear. If any template argument is
  /// not supported assert is created and the argument omitted.
  ///
  /// \param SuffixOut Output stream where encoded suffix will be written to.
  /// \param TArg      Template argument in demangler's result to encode in
  ///                  new data name (as optional suffix).
  /// \param ScanPacks Indicates that nested template parameter packs should
  ///                  also be scanned.
  /// \return          Value indicating that encoding was successful.
  static bool encodeDataTypeTArgs(printer::StringOStreamT &SuffixOut,
                                       const DmngRsltTArg &TArg,
                                       bool ScanPacks = false) {
    if (ScanPacks && TArg.isPack()) {
      SuffixOut << encodeDataTypeTArgs(TArg.getPack(), ScanPacks);
      return true;
    }

    // Encode supported types specified as template argument.
    auto EffectiveType = getEffectiveType(TArg);
    if (EffectiveType != nullptr) {
      auto TypeEncoding = getSpecialReturnTypeEncoding(EffectiveType, true);
      if (TypeEncoding.empty())
        return false;

      SuffixOut << "_" << TypeEncoding;
      return true;
    }

    // Resolve literal expression with non-void content.
    auto EffectiveExpr = getEffectivePrimaryExpr(TArg);
    if (EffectiveExpr == nullptr || !EffectiveExpr->isLiteral())
      return false;

    // Get content as integral. Negative integral is encoded as 64-bit unsigned
    // value (no bit change, as its value was incremented by 2^64). Boolean
    // value is encoded as 0 (false) or 1 (true).
    // For negative integral value assert is generated.
    switch (EffectiveExpr->getContentType()) {
    case DmngRsltPrimaryExpr::UInt:
    case DmngRsltPrimaryExpr::Bool:
      SuffixOut << "_" << EffectiveExpr->getContentAsUInt();
      return true;
    case DmngRsltPrimaryExpr::SInt:
      assert(EffectiveExpr->getContentAsSInt() >= 0 &&
             "Encoding of negative value is not really supported.");
      SuffixOut << "_" << EffectiveExpr->getContentAsUInt();
      return true;
    default:
      return false;
    }
  }

  /// \brief Encodes template arguments of data type (or data variable).
  ///
  /// Template arguments are encoded in order they appear on template argument
  /// list. If parameter packs are also scanned, they are flattened. If any
  /// template argument is not supported assert is created and the argument
  /// omitted.
  ///
  /// \param SuffixOut Output stream where encoded suffix will be written to.
  /// \param TArgs     Template arguments collection in demangler's result
  ///                  to encode in new data name (as optional suffix).
  /// \param ScanPacks Indicates that nested template parameter packs should
  ///                  also be scanned.
  /// \return          Encoded name of special enum value, or empty string if
  ///                  template arguments cannot be encoded or there is no
  ///                  template arguments to encode.
  static std::string encodeDataTypeTArgs(
      const DmngRsltTArg::TArgsPackColT &TArgs, bool ScanPacks = false) {
    if (TArgs.empty()) // Fast return.
      return std::string();

    // NOTE: Helper string to support llvm::raw_string_ostream.
    std::string Suffixes;
    printer::StringOStreamT SuffixesStream(Suffixes);

    for (const auto &TArg : TArgs) {
      bool Success = encodeDataTypeTArgs(SuffixesStream, TArg, ScanPacks);
      assert(Success && "Template argument is not supported for encoding.");
    }

    return SuffixesStream.str();
  }


  /// \brief Determines whether result entity (node or template argument) is
  ///        type/value/instance-dependant.
  ///
  /// \param TArg Tested template argument.
  /// \return     true if argument is dependant (it references template
  ///             argument from outer scope); otherwise, false.
  static bool isDependant(const DmngRsltTArg &TArg) {
    if (TArg.isType())
      return isDependant(TArg.getType());
    if (TArg.isExpression())
      return isDependant(TArg.getExpression());
    if (TArg.isPack()) {
      for (const auto &PackArg : TArg.getPack()) {
        if (isDependant(PackArg))
          return true;
      }
    }
    return false;
  }

  /// \brief Determines whether result entity (node or template argument) is
  ///        type/value/instance-dependant.
  ///
  /// \param NameNode Tested name node from demangler result.
  /// \param TArgScopeChanged Placeholder for boolean value indicating that
  ///                         scope of template arguments has changed (only
  ///                         set on change - pass variable initialized to
  ///                         false). All template referencing after scope
  ///                         change referes to new scope.
  /// \return                 true if argument is dependant (it references
  ///                         template argument from outer scope); otherwise,
  ///                         false.
  static bool isDependant(const std::shared_ptr<const DmngRsltName> &NameNode,
                          bool *TArgScopeChanged = nullptr) {
    if (NameNode == nullptr)
      return false;

    auto SpecName = NameNode->getAs<DNK_Special>();
    if (SpecName != nullptr) {
      return isDependant(SpecName->getRelatedObject()) ||
               isDependant(SpecName->getOrigin()) ||
               isDependant(SpecName->getRelatedObject());
    }

    auto OrdName = NameNode->getAs<DNK_Ordinary>();
    if (OrdName != nullptr) {
      for (const auto &VQual : OrdName->getVendorQuals()) {
        if (!VQual.isTemplate())
          continue;
        for (const auto &TArg : VQual.getTemplateArgs()) {
          if (isDependant(TArg))
            return true;
        }
      }

      bool ScopeChanged = false;
      if (isDependant(OrdName->getLocalScope(), &ScopeChanged))
        return true;
      if (ScopeChanged) {
        if (TArgScopeChanged != nullptr)
          *TArgScopeChanged = true;
        return false;
      }

      for (const auto &Part : OrdName->getParts()) {
        if (Part->getPartKind() == DNPK_TemplateParam)
          return true;
        if (Part->getPartKind() == DNPK_Decltype &&
            isDependant(Part->getAs<DNPK_Decltype>()->getDecltype())) {
          return true;
        }

        if (!Part->isTemplate())
          continue;
        for (const auto &TArg : Part->getTemplateArgs()) {
          if (isDependant(TArg))
            return true;
        }
        ScopeChanged = true;
      }
      if (ScopeChanged) {
        if (TArgScopeChanged != nullptr)
          *TArgScopeChanged = true;
        return false;
      }

      for (const auto &SigType : OrdName->getSignatureTypes()) {
        if (isDependant(SigType))
          return true;
      }
    }

    return false;
  }

  /// \brief Determines whether result entity (node or template argument) is
  ///        type/value/instance-dependant.
  ///
  /// \param ExprNode Tested expression node from demangler result.
  /// \return         true if argument is dependant (it references template
  ///                 argument from outer scope); otherwise, false.
  static bool isDependant(const std::shared_ptr<const DmngRsltExpr> &ExprNode) {
    if (ExprNode == nullptr)
      return false;

    switch (ExprNode->getKind()) {
    case DXK_TemplateParam:
      return true;
    case DXK_Decltype:
      return isDependant(ExprNode->getAs<DXK_Decltype>()->getExpression());
    case DXK_Primary:
      return isDependant(ExprNode->getAs<DXK_Primary>()->getExternalName());
    default:
      return false;
    }
  }

  /// \brief Determines whether result entity (node or template argument) is
  ///        type/value/instance-dependant.
  ///
  /// \param TypeNode Tested type node from demangler result.
  /// \return         true if argument is dependant (it references template
  ///                 argument from outer scope); otherwise, false.
  static bool isDependant(const std::shared_ptr<const DmngRsltType> &TypeNode) {
    if (TypeNode == nullptr)
      return false;

    switch (TypeNode->getKind()) {
    case DTK_Builtin:
      return false;
    case DTK_TemplateParam:
    case DTK_PackExpansion: // Pack expansion always requires template param.
      return true;
    case DTK_Function: {
        auto FuncType = TypeNode->getAs<DTK_Function>();
        for (const auto &SigType : FuncType->getSignatureTypes()) {
          if (isDependant(SigType))
            return true;
        }
        return false;
      }
    case DTK_TypeName:
      return isDependant(TypeNode->getAs<DTK_TypeName>()->getTypeName());
    case DTK_Array: {
        auto ArrayType = TypeNode->getAs<DTK_Array>();
        return isDependant(ArrayType->getSizeExpr()) ||
                 isDependant(ArrayType->getElemType());
      }
    case DTK_Vector: {
        auto VecType = TypeNode->getAs<DTK_Vector>();
        return isDependant(VecType->getSizeExpr()) ||
                 isDependant(VecType->getElemType());
      }
    case DTK_PointerToMember: {
        auto P2MType = TypeNode->getAs<DTK_PointerToMember>();
        return isDependant(P2MType->getClassType()) ||
                 isDependant(P2MType->getMemberType());
      }
    case DTK_Decltype:
      return isDependant(TypeNode->getAs<DTK_Decltype>()->getDecltype());
    case DTK_Pointer:
      return isDependant(TypeNode->getAs<DTK_Pointer>()->getInnerType());
    case DTK_LValueRef:
      return isDependant(TypeNode->getAs<DTK_LValueRef>()->getInnerType());
    case DTK_RValueRef:
      return isDependant(TypeNode->getAs<DTK_RValueRef>()->getInnerType());
    case DTK_C2000Complex:
      return isDependant(
        TypeNode->getAs<DTK_C2000Complex>()->getInnerType());
    case DTK_C2000Imaginary:
      return isDependant(
        TypeNode->getAs<DTK_C2000Imaginary>()->getInnerType());
    case DTK_QualGroup: {
        auto QType = TypeNode->getAs<DTK_QualGroup>();
        // Most likely case first (dependant type is inner type).
        if (isDependant(QType->getInnerType()))
          return true;

        for (const auto &VQual : QType->getVendorQuals()) {
          if (!VQual.isTemplate())
            continue;
          for (const auto &TArg : VQual.getTemplateArgs()) {
            if (isDependant(TArg))
              return true;
          }
        }

        return false;
      }
    default:
      return false;
    }
  }


  /// \brief Adapts input name node which identifies data type/variable node.
  ///
  /// The method tries to adapt data/variable name node. If node is not data
  /// node, the method fails adaptation.
  ///
  /// \param DataNode     Input data node which will be adapted. Node must
  ///                     exist, otherwise assert is generated.
  /// \param ExtSetBiName Triple that identifies extended set, built-in name and
  ///                     its category. This triple is result of getExtSetBiName
  ///                     method. If the triple is invalid, the method asserts.
  /// \return             Adaptation result pair. First element indicates
  ///                     status of adaptation. Second contains possibly adapted
  ///                     node. For details please see "Adaptation traits"
  ///                     paragraph ItaniumEncodeTraits.
  static std::pair<printer::AdaptResult,
                   std::shared_ptr<const DmngRsltOrdinaryName>>
  adaptDataTypeName(
      const std::shared_ptr<const DmngRsltOrdinaryName> &DataNode,
      const std::tuple<std::string, std::string, BiCategory> &ExtSetBiName) {
    assert(DataNode != nullptr && "Node must exist.");
    assert(std::get<2>(ExtSetBiName) != BiCategory::Failure &&
           "Built-in name and category must be valid.");

    // If name does not name data (it is function name), fail adaptation.
    if (!DataNode->isData())
      return std::make_pair(printer::AR_Failure, DataNode);

    // If there are no namespace (cl::__spriv). The node was already adapted.
    if (DataNode->getParts().size() <= 2)
      return std::make_pair(printer::AR_NoAdapt, DataNode);

    // Check whether built-in category is data type or variable. Reflow
    // only matching categories.
    if (std::get<2>(ExtSetBiName) != BiCategory::Type &&
        std::get<2>(ExtSetBiName) != BiCategory::Variable)
      return std::make_pair(printer::AR_NoAdapt, DataNode);

    // Generate data type name's or variable's optional prefixes.
    std::string OptionalPrefixes;
    if (std::get<2>(ExtSetBiName) == BiCategory::Variable)
      OptionalPrefixes = "BuiltIn";

    // Generate data type name's or variable's optional suffixes for special
    // values and types.
    std::string OptionalSuffixes;
    if (DataNode->getLastPart()->isTemplate()) {
      OptionalSuffixes = encodeDataTypeTArgs(
                           DataNode->getLastPart()->getTemplateArgs());
      if (!OptionalSuffixes.empty())
        OptionalSuffixes.insert(0, "_");
    }

    // Create new built-in name (data).
    std::string NewBiName = "__spirv_";
    if (!std::get<0>(ExtSetBiName).empty())
      NewBiName += std::get<0>(ExtSetBiName) + "_";
    NewBiName += OptionalPrefixes + std::get<1>(ExtSetBiName) +
                   OptionalSuffixes;

    // Create new node representing new data type/variable name.
    auto NewNode = DataNode->clone(false);
    NewNode->setPart(std::make_shared<DmngRsltSrcNamePart>(NewBiName));
    NewNode->resetQuals();

    return std::make_pair(printer::AR_Adapt, NewNode);
  }


public:
  /// \brief Adapts name node (special).
  ///
  /// For details please see "Adaptation traits" paragraph ItaniumEncodeTraits.
  std::pair<printer::AdaptResult, std::shared_ptr<const DmngRsltOrdinaryName>>
  adapt(const std::shared_ptr<const DmngRsltSpecialName> &Node) const {
    assert(Node != nullptr && "Node must exist.");

    if (Node->getSpecialKind() != DSNK_TypeInfoNameString)
      return std::make_pair(printer::AR_Failure, nullptr);
    auto RelTypeNameNode = Node->getRelatedType()->getAs<DTK_TypeName>();
    if (RelTypeNameNode == nullptr)
      return std::make_pair(printer::AR_Failure, nullptr);

    return std::make_pair(printer::AR_Adapt, RelTypeNameNode->getTypeName());
  }

  /// \brief Adapts name node (ordinary).
  ///
  /// For details please see "Adaptation traits" paragraph ItaniumEncodeTraits.
  std::pair<printer::AdaptResult, std::shared_ptr<const DmngRsltOrdinaryName>>
  adapt(const std::shared_ptr<const DmngRsltOrdinaryName> &Node) const {
      assert(Node != nullptr && "Node must exist.");

      // If there are no namespace (cl::__spriv). The node was already adapted.
      if (Node->getParts().size() <= 2)
        return std::make_pair(printer::AR_NoAdapt, Node);

      // Detect extended set and built-in names.
      auto ExtSetBiName = getExtSetBiName(Node);
      if (std::get<2>(ExtSetBiName) == BiCategory::Failure)
        return std::make_pair(printer::AR_Failure, Node);

      // Handle data type names.
      if (Node->isData())
        return adaptDataTypeName(Node, ExtSetBiName);

      // Generate function's optional suffixes for special values and types.
      std::string OptionalSuffixes;
      if (Node->getLastPart()->isTemplate()) {
        if (Node->hasReturnType() && isDependant(Node->getReturnType())) {
          auto SpecRet = getSpecialReturnTypeEncoding(Node->getReturnType());
          assert(!SpecRet.empty() &&
                 "Template has dependant return type, but it is not special "
                 "return type. There is currently no way to encode it.");
          if (!SpecRet.empty())
            OptionalSuffixes += "_R" + SpecRet;
        }

        auto SpecSatEnum = getSpecialEnumValueEncoding<SpecialEnumType::Sat>(
                              Node->getLastPart()->getTemplateArgs());
        if (SpecSatEnum != nullptr)
          OptionalSuffixes += SpecSatEnum;

        auto SpecRmEnum =
          getSpecialEnumValueEncoding<SpecialEnumType::RoundingMode>(
            Node->getLastPart()->getTemplateArgs());
        if (SpecRmEnum != nullptr)
          OptionalSuffixes += SpecRmEnum;
      }

      // Create new built-in name (function).
      std::string NewBiName = "__spirv_";
      if (!std::get<0>(ExtSetBiName).empty()) {
        NewBiName += std::get<0>(ExtSetBiName) + "_";
        if (!OptionalSuffixes.empty())
          OptionalSuffixes.insert(0, "_");
      }
      NewBiName += std::get<1>(ExtSetBiName) + OptionalSuffixes;

      // Create new node representing new function name.
      auto NewNode = Node->clone(false);
      NewNode->setPart(std::make_shared<DmngRsltSrcNamePart>(NewBiName));
      NewNode->resetQuals();
      // Node is not template anymore, so encoded return type need to be
      // removed.
      if (Node->hasReturnType()) {
        NewNode->resetSignature();
        for (const auto &ParamType : Node->getParamTypes())
          NewNode->addSignatureType(ParamType);
      }

      return std::make_pair(printer::AR_Adapt, NewNode);
  }

  /// \brief Adapts array and vector nodes (type flattening).
  ///
  /// For details please see "Adaptation traits" paragraph ItaniumEncodeTraits.
  std::pair<printer::AdaptResult, std::shared_ptr<const DmngRsltArrayVecType>>
  adapt(const std::shared_ptr<const DmngRsltArrayVecType> &Node) const {
    assert(Node != nullptr && "Node must exist.");

    // Array or vector has already got effective size.
    if (!Node->isSizeSpecified() || Node->getSizeExpr() == nullptr)
      return std::make_pair(printer::AR_NoAdapt, Node);

    // If effective size cannot be computed, do not adapt array/vector.
    auto EffectiveSize = getEffectiveArrayVecElements(Node);
    if (EffectiveSize == 0)
      return std::make_pair(printer::AR_NoAdapt, Node);

    auto NewArrayVec = std::make_shared<DmngRsltArrayVecType>(
                         Node->getElemType(), EffectiveSize,
                         Node->getKind() == DTK_Vector);
    return std::make_pair(printer::AR_Adapt, NewArrayVec);
  }
};

struct OclCxxBifNameReflower : llvm::ModulePass {
  /// Identifier of the pass (its address).
  static char ID;

  OclCxxBifNameReflower() : ModulePass(ID) {}

  //StringRef getPassName() const override { return "oclcxx-bif-name-reflower"; }

  bool runOnModule(llvm::Module &M) override {
    LLVM_DEBUG(llvm::dbgs() << "\n\nSTART OCLCXX-BIF-NAME-REFLOWER PASS\n");
    using EncodeTraits = printer::ItaniumEncodeTraits<OclCxxBiAdaptTraits,
                                                      true, true, true, true>;

    bool Modified = false;
    ItaniumNameParser Parser(OclASExtract);

    // Process type names.
    for (llvm::StructType *T : M.getIdentifiedStructTypes()) {
      if (T == nullptr)
        continue;
      auto TypeName = T->getName();
      if (TypeName.empty())
        continue;

      // Extract mangled part of LLVM type name.
      auto TypeNameMStart = TypeName.find('.');
      if (TypeNameMStart == llvm::StringRef::npos)
        continue;
      ++TypeNameMStart;
      auto TypeNameMEnd = TypeName.find('.', TypeNameMStart);
      if (TypeNameMEnd == llvm::StringRef::npos)
        continue;

      auto MangledName = TypeName.substr(TypeNameMStart,
                                         TypeNameMEnd - TypeNameMStart).str();
      // Original LLVM type name (before addition of mangled part).
      auto OriginalName = TypeName.substr(0, TypeNameMStart).str() +
                          TypeName.substr(TypeNameMEnd + 1).str();

      // Reflow name of LLVM type (if possible) or restore original.
      if (MangledName.empty()) {
        T->setName(OriginalName); // restore original.
        continue;
      }

      auto PResult = Parser.parse(MangledName);
      if (!OclCxxBiAdaptTraits::isCandidateForReflow(PResult, true)) {
        T->setName(OriginalName); // restore original.
        continue;
      }

      // NOTE: Helper string to support llvm::raw_string_ostream.
      EncodeTraits::StringT NewTypeName;
      EncodeTraits::StringOStreamT NewTypeNameStream(NewTypeName);

      if (!printer::print<EncodeTraits>(NewTypeNameStream, PResult).first) {
        T->setName(OriginalName); // restore original.
        continue;
      }

      // Check if has proper prefix in name.
      NewTypeName = NewTypeNameStream.str();
      if (NewTypeName.compare(0, 8, "__spirv_") != 0) {
        T->setName(OriginalName); // restore original.
        continue;
      }

      // Reformat name string to form correct for LLVM type elements.
      auto ResultTypeName = "spirv." + NewTypeName.substr(8);
      auto PostfixPos = ResultTypeName.find("__");
      if (PostfixPos != std::string::npos)
        ResultTypeName[PostfixPos] = '.';

      LLVM_DEBUG(llvm::dbgs() << "\nOCLCXX-BIF-NAME-REFLOWER: "
                              << "The TYPE name is fixed from:\n  " << TypeName
                              << "\n to:\n  " << ResultTypeName << "\n");

      T->setName(ResultTypeName);
      Modified = true;
    }

    // Process global variables.
    for (llvm::GlobalVariable &G : M.globals()) {
      auto PResult = Parser.parse(G.getName());
      if (!OclCxxBiAdaptTraits::isCandidateForReflow(PResult))
        continue;

      // NOTE: Helper string to support llvm::raw_string_ostream.
      EncodeTraits::StringT NewBiName;
      EncodeTraits::StringOStreamT NewBiNameStream(NewBiName);

      if (printer::print<EncodeTraits>(NewBiNameStream, PResult).first) {
        G.setName(NewBiNameStream.str());
        Modified = true;
      }
    }

    // Process function names.
    for (llvm::Function &F : M) {
      auto PResult = Parser.parse(F.getName());
      if (!OclCxxBiAdaptTraits::isCandidateForReflow(PResult))
        continue;

      // NOTE: Helper string to support llvm::raw_string_ostream.
      EncodeTraits::StringT NewBiName;
      EncodeTraits::StringOStreamT NewBiNameStream(NewBiName);

      if (printer::print<EncodeTraits>(NewBiNameStream, PResult).first) {
        LLVM_DEBUG(llvm::dbgs() << "\nOCLCXX-BIF-NAME-REFLOWER: "
                                << "The FUNC name has been fixed from:\n  "
                                << F.getName() << "\nto:\n  "
                                << NewBiNameStream.str() << "\n");
        F.setName(NewBiNameStream.str());
        Modified = true;
      }
    }
    return Modified;
  }
};
}


char OclCxxBifNameReflower::ID = 0;

using namespace llvm;
INITIALIZE_PASS(OclCxxBifNameReflower,
                "oclcxx-bif-name-reflower",
                "Built-in name reflower for OpenCL C++",
                false, false)

ModulePass *llvm::createOclCxxBifNameReflowerPass() {
  return new OclCxxBifNameReflower;
}
