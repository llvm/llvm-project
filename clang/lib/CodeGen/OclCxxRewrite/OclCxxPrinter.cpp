//===- OclCxxPrinter.cpp - OCLC++ type/name printer & mangler   -*- C++ -*-===//
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


#include "OclCxxPrinter.h"
#include "OclCxxDemangler.h"

#include <cassert>
#include <type_traits>
#include <utility>

#if! OCLCXXREWRITE_PRINTER_USE_LLVM_STREAMS
#include <sstream>
#endif

using namespace oclcxx::adaptation;
using namespace printer;

// -----------------------------------------------------------------------------
// HELPERS FOR PRINTER / ENCODER
// -----------------------------------------------------------------------------
// Node / demangler result tools.

/// \brief Indicates that node with type description describes "void" type.
///
/// \param TypeNode Node with type description.
/// \return         true if node describes "void" type; otherwise, false.
inline static bool isVoidType(
  const std::shared_ptr<const DmngRsltType> &TypeNode) {
  if (TypeNode == nullptr || TypeNode->getKind() != DTK_Builtin)
    return false;
  return TypeNode->getAs<DTK_Builtin>()->getBuiltinType() == DBT_Void;
}

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF C++-LIKE ENCODER / PRINTER TRAITS
// -----------------------------------------------------------------------------

/// \def OCLCXX_CLPT_ENCODE(...)
/// Invokes encode function overload for specified node with short circuiting
/// when encode fails.
///
/// Parameters are the same as for encode() function.
#define OCLCXX_CLPT_ENCODE(...)                                                \
do {                                                                           \
  if (encode(__VA_ARGS__) == ER_Failure)                                       \
    return ER_Failure;                                                         \
} while(false)

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

CxxLikeEncodeTraits::OStreamWrapperT CxxLikeEncodeTraits::createStreamWrapper(
    StringOStreamT &Out) {
  return OStreamWrapperT(Out);
}

CxxLikeEncodeTraits::StringT CxxLikeEncodeTraits::processResult(
    const StringT &Result) {
  StringT CorrectedResult;
  CorrectedResult.reserve(Result.length());

  for (auto I = Result.cbegin(), E = Result.cend(); I != E; ++I) {
    CorrectedResult.push_back(*I);

    if (*I == '(' || *I == '<' || *I == ' ') {
      ++I;
      while (I != E && *I == ' ')
        ++I;
      --I;
    }
  }

  return CorrectedResult;
}

// -----------------------------------------------------------------------------

EncodeResult CxxLikeEncodeTraits::encodeResult(
    OStreamWrapperT &Out, const std::shared_ptr<const DmngRsltNode> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  switch (Value->getNodeKind()) {
  case DNDK_Name:
    return encode(Out, Value->getAs<DNDK_Name>());
  case DNDK_NamePart:
    return encode(Out, Value->getAs<DNDK_NamePart>());
  case DNDK_Type:
    return encode(Out, Value->getAs<DNDK_Type>());
  case DNDK_Expr:
    return encode(Out, Value->getAs<DNDK_Expr>());
  case DNDK_NameParts:
    // ReSharper disable once CppUnreachableCode
    assert(false && "Printer does not support printing of name parts helper.");
    return ER_Failure;
  default:
    // ReSharper disable once CppUnreachableCode
    assert(false && "Printer does not support current node kind.");
    return ER_Failure;
  }
}

EncodeResult CxxLikeEncodeTraits::encodeResult(OStreamWrapperT &Out,
                                               const DmngRslt &Result) {
  if (Result.isFailed() || Result.getName() == nullptr)
    return ER_Failure;

  return encode(Out, Result.getName());
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Non-node types.

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out, const DmngRsltVendorQual &Value) {
  Out << "dmng::ext_qual(" << Value.getName();

  if (encodeTArgs(Out, Value) == ER_Failure)
    return ER_Failure;

  Out << ")";
  return ER_Success;
}

// -----------------------------------------------------------------------------

// R-L order, reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out, const DmngRsltTArg &Value) {
  if (Value.isType())
    return encode(Out, Value.getType()); // {SCOPE NESTING}

  if (Value.isExpression()) {
    // Assembling change (reversed -> ordinary).
    auto ExprOut = Out.createPrefixChildStream();
    return encode(ExprOut, Value.getExpression());
  }

  if (Value.isPack()) {
    // Assembling change (reversed -> ordinary).
    auto PackOut = Out.createPrefixChildStream();
    PackOut << "pack``{";
    const char *Sep = "";
    for (const auto &TArg : Value.getPack()) {
      PackOut << Sep;
      // Assembling change (ordinary -> reversed).
      auto TArgOut = PackOut.createChildStream();
      OCLCXX_CLPT_ENCODE(TArgOut, TArg);
      Sep = ", ";
    }
    PackOut << "}";

    return ER_Success;
  }

  return ER_Failure;
}

// R-L order, reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out, const std::shared_ptr<const DmngRsltTArg> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  return encode(Out, *Value); // {SCOPE NESTING}
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out, const DmngRsltAdjustOffset &Value) {
  if (Value.isVirtual()) {
    Out << "vbase(offset: "
        << asHex<StringT, true, true>(Value.getBaseOffset()) << ", vcall: "
        << asHex<StringT, true, true>(Value.getVCallOffset()) << ")";
  }
  else {
    Out << "base(offset: "
        << asHex<StringT, true, true>(Value.getBaseOffset()) << ")";
  }

  return ER_Success;
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out, const DmngCvrQuals &Value) {
  if (Value & DCVQ_Const)
    Out << " const";
  if (Value & DCVQ_Volatile)
    Out << " volatile";
  if (Value & DCVQ_Restrict)
    Out << " __restrict__";

  return ER_Success;
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out, const DmngRefQuals &Value) {
  switch (Value) {
  case DRQ_None:      break;
  case DRQ_LValueRef: Out << " &"; break;
  case DRQ_RValueRef: Out << " &&"; break;
  default:
    // ReSharper disable once CppUnreachableCode
    assert(false && "Printer does not support current reference qualifier.");
    return ER_Failure;
  }

  return ER_Success;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Non-node abstract bases.

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encodeVendorQuals(
    OStreamWrapperT &Out, const DmngRsltVendorQualsBase &Value) {
  switch (Value.getAsQuals()) {
  case DASQ_None:     break;
  case DASQ_Private:  Out << " __private"; break;
  case DASQ_Local:    Out << " __local"; break;
  case DASQ_Global:   Out << " __global"; break;
  case DASQ_Constant: Out << " __constant"; break;
  case DASQ_Generic:  Out << " __generic"; break;
  default:
    // ReSharper disable once CppUnreachableCode
    assert(false && "Printer does not support current addressspace qualifier.");
    return ER_Failure;
  }

  if (Value.hasVendorQuals()) {
    Out << " [[";
    const char *Sep = "";
    for (const auto &VQual : Value.getVendorQuals()) {
      Out << Sep;
      OCLCXX_CLPT_ENCODE(Out, VQual);
      Sep = ", ";
    }
    Out << "]]";
  }

  return ER_Success;
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encodeTArgs(
    OStreamWrapperT &Out, const DmngRsltTArgsBase &Value) {
  if (Value.isTemplate()) {
    Out << "<";
    const char *Sep = "";
    for (const auto &TArg : Value.getTemplateArgs()) {
      Out << Sep;
      // Assembling change (ordinary -> reversed).
      auto TArgOut = Out.createChildStream();
      OCLCXX_CLPT_ENCODE(TArgOut, TArg);
      Sep = ", ";
    }
    Out << ">";
  }

  return ER_Success;
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encodeSignatureParams(
    OStreamWrapperT &Out, const DmngRsltSignatureTypesBase &Value) {
  if (!Value.getSignatureTypes().empty()) {
    Out << "(";
    const char *Sep = "";
    for (const auto &ParamType : Value.getParamTypes()) {
      if (!isVoidType(ParamType)) {
        Out << Sep;
        // Assembling change (ordinary -> reversed).
        auto ParamOut = Out.createChildStream();
        OCLCXX_CLPT_ENCODE(ParamOut, ParamType);
        Sep = ", ";
      }
    }
    Out << ")";
  }

  return ER_Success;
}

// -----------------------------------------------------------------------------

// R-L order, reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encodeSignatureReturn(
    OStreamWrapperT &Out, const DmngRsltSignatureTypesBase &Value) {
  if (Value.hasReturnType())
    return encode(Out, Value.getReturnType()); // {SCOPE NESTING}

  auto TypeOut = Out.createPrefixChildStream();
  TypeOut << "???";

  return ER_Success;
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encodeNameParts(
    OStreamWrapperT &Out, const DmngRsltNamePartsBase &Value) {
  const char *Sep = "";
  std::shared_ptr<const DmngRsltNamePart> PrevPart;
  for (const auto &NamePart : Value.getParts()) {
    Out << Sep;
    if (NamePart->isDataMember())
      Out << "data``{";
    OCLCXX_CLPT_ENCODE(Out, NamePart, PrevPart);
    if (NamePart->isDataMember())
      Out << "}";
    Sep = "::";

    PrevPart = NamePart;
  }

  return ER_Success;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Nodes (expression).

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out, const std::shared_ptr<const DmngRsltExpr> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  switch (Value->getKind()) {
  case DXK_Decltype:
    return encode(Out, Value->getAs<DXK_Decltype>());
  case DXK_TemplateParam:
    return encode(Out, Value->getAs<DXK_TemplateParam>());
  case DXK_Primary:
    return encode(Out, Value->getAs<DXK_Primary>());
  default:
    // ReSharper disable once CppUnreachableCode
    assert(false && "Printer does not support current node kind.");
    return ER_Failure;
  }
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltDecltypeExpr> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  Out << "decltype(";
  OCLCXX_CLPT_ENCODE(Out, Value->getExpression());
  Out << ")";
  if (Value->isSimple())
    Out << " [[dmng::simple]]";

  return ER_Success;
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
// or
// R-L order, reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltTParamExpr> &Value,
    bool ReverseAssembly) {
  if (Value == nullptr)
    return ER_Failure;

  if (ReverseAssembly) {
    auto TParamOut = Out.createPrefixChildStream();
    TParamOut << " [[dmng::tparam(" << Value->getReferredTArgIdx() << ")]]";

    return encode(Out, Value->getReferredTArg()); // {SCOPE NESTING}
  }

  // Assembling change (ordinary -> reversed).
  auto TArgOut = Out.createChildStream();
  OCLCXX_CLPT_ENCODE(TArgOut, Value->getReferredTArg());
  Out << " [[dmng::tparam(" << Value->getReferredTArgIdx() << ")]]";

  return ER_Success;
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltPrimaryExpr> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  if (Value->isExternalName()) {
    // Assembling change (ordinary -> reversed).
    auto ExternNameOut = Out.createChildStream();
    OCLCXX_CLPT_ENCODE(ExternNameOut, Value->getExternalName());
  } else if (Value->isLiteral()) {
    Out << "(";
    // Assembling change (ordinary -> reversed).
    auto LiteralTypeOut = Out.createChildStream();
    OCLCXX_CLPT_ENCODE(LiteralTypeOut, Value->getLiteralType());
    Out << ")";

    switch (Value->getContentType()) {
    case DmngRsltPrimaryExpr::Void:
      break;
    case DmngRsltPrimaryExpr::UInt:
      Out << " " << Value->getContentAsUInt();
      break;
    case DmngRsltPrimaryExpr::SInt:
      Out << " " << Value->getContentAsSInt();
      break;
    case DmngRsltPrimaryExpr::Bool:
      Out << " " << (Value->getContentAsBool() ? "true" : "false");
      break;
    default:
      // ReSharper disable once CppUnreachableCode
      assert(false && "Printer does not support literal content type.");
      return ER_Failure;
    }
  }

  return ER_Success;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Nodes (type).

// R-L order, reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out, const std::shared_ptr<const DmngRsltType> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  // {SCOPE NESTING}
  switch (Value->getKind()) {
  case DTK_Builtin:
    return encode(Out, Value->getAs<DTK_Builtin>());
  case DTK_Function:
    return encode(Out, Value->getAs<DTK_Function>());
  case DTK_TypeName:
    return encode(Out, Value->getAs<DTK_TypeName>());
  case DTK_Array:
    return encode(Out, Value->getAs<DTK_Array>());
  case DTK_Vector:
    return encode(Out, Value->getAs<DTK_Vector>());
  case DTK_PointerToMember:
    return encode(Out, Value->getAs<DTK_PointerToMember>());
  case DTK_TemplateParam:
    return encode(Out, Value->getAs<DTK_TemplateParam>());
  case DTK_Decltype:
    return encode(Out, Value->getAs<DTK_Decltype>());
  case DTK_Pointer:
    return encode(Out, Value->getAs<DTK_Pointer>());
  case DTK_LValueRef:
    return encode(Out, Value->getAs<DTK_LValueRef>());
  case DTK_RValueRef:
    return encode(Out, Value->getAs<DTK_RValueRef>());
  case DTK_C2000Complex:
    return encode(Out, Value->getAs<DTK_C2000Complex>());
  case DTK_C2000Imaginary:
    return encode(Out, Value->getAs<DTK_C2000Imaginary>());
  case DTK_PackExpansion:
    return encode(Out, Value->getAs<DTK_PackExpansion>());
  case DTK_QualGroup:
    return encode(Out, Value->getAs<DTK_QualGroup>());
  default:
    // ReSharper disable once CppUnreachableCode
    assert(false && "Printer does not support current node kind.");
    return ER_Failure;
  }
}

// -----------------------------------------------------------------------------

// R-L order, reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltBuiltinType> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  auto TypeOut = Out.createPrefixChildStream();
  if (Value->isVendorBuiltinType())
    TypeOut << Value->getVendorName() << " [[dmng::ext_type]]";
  else
    TypeOut << getFixedBuiltinTypeName(Value->getBuiltinType());

  return ER_Success; // {SCOPE LEAF}
}

// -----------------------------------------------------------------------------

// R-L order on return type / L-R order on parameters and qualifiers,
// reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltFuncType> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  auto IsOutPure = Out.isPure();
  auto FuncTypeOut = Out.createPrefixChildStream();
  FuncTypeOut << (IsOutPure ? " " : " (");

  // {PARENT SCOPE}

  Out << (IsOutPure ? "" : ")");
  if (encodeSignatureParams(Out, *Value) == ER_Failure)
    return ER_Failure;
  // TODO: Add transactional qualifier in the future (Dx).
  if (Value->isExternC())
    Out << " [[dmng::extern(\"C\")]]";
  if (encodeVendorQuals(Out, *Value) == ER_Failure)
    return ER_Failure;
  OCLCXX_CLPT_ENCODE(Out, Value->getCvrQuals());
  OCLCXX_CLPT_ENCODE(Out, Value->getRefQuals());

  return encodeSignatureReturn(Out, *Value); // {SCOPE NESTING}
}

// -----------------------------------------------------------------------------

// R-L order, reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltTypeNameType> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  // {SCOPE PREFIX}
  auto TypeNameOut = Out.createPrefixChildStream();
  switch (Value->getElaboration()) {
  case DTNK_None:            break;
  case DTNK_ElaboratedClass: TypeNameOut << "class "; break;
  case DTNK_ElaboratedUnion: TypeNameOut << "union "; break;
  case DTNK_ElaboratedEnum:  TypeNameOut << "enum "; break;
  default:
    // ReSharper disable once CppUnreachableCode
    assert(false && "Printer does not support current elaboration.");
    return ER_Failure;
  }

  OCLCXX_CLPT_ENCODE(TypeNameOut, Value->getTypeName()); // {SCOPE NESTING}

  return ER_Success;
}

// -----------------------------------------------------------------------------

// R-L order on array elem type (L-R order in rest), reversed assembling,
// separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltArrayVecType> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  // Vector encoding has ordinary assembling.
  if (Value->getKind() == DTK_Vector) {
    // Assembling change (reversed -> ordinary).
    auto VecTypeOut = Out.createPrefixChildStream();

    VecTypeOut << "vec``{";
    // Assembling change (ordinary -> reversed).
    auto ElemTypeOut = VecTypeOut.createChildStream();
    OCLCXX_CLPT_ENCODE(ElemTypeOut, Value->getElemType());

    if (Value->isSizeSpecified()) {
      VecTypeOut << ", ";
      if (Value->getSizeExpr() != nullptr)
        OCLCXX_CLPT_ENCODE(VecTypeOut, Value->getSizeExpr());
      else
        VecTypeOut << Value->getSize();
    }
    VecTypeOut << "}";

    return ER_Success;
  }

  auto IsOutPure = Out.isPure();
  auto ArrayTypeOut = Out.createPrefixChildStream();
  ArrayTypeOut << (IsOutPure ? " " : " (");

  // {PARENT SCOPE}

  Out << (IsOutPure ? "[" : ")[");
  if (Value->isSizeSpecified()) {
    if (Value->getSizeExpr() != nullptr)
      OCLCXX_CLPT_ENCODE(Out, Value->getSizeExpr());
    else
      Out << Value->getSize();
  }
  Out << "]";

  return encode(Out, Value->getElemType()); // {SCOPE NESTING}
}

// -----------------------------------------------------------------------------

// R-L order, reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltPtr2MmbrType> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  auto ClassNameOut = Out.createPrefixChildStream();
  ClassNameOut << " ";
  auto ClassTypeOut = ClassNameOut.createChildStream();
  OCLCXX_CLPT_ENCODE(ClassTypeOut, Value->getClassType());
  ClassNameOut << " :: *";

  return encode(Out, Value->getMemberType()); // {SCOPE NESTING}
}

// -----------------------------------------------------------------------------

// R-L order, reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltTParamType> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  // Assembling change (reversed -> ordinary).
  auto TParamOut = Out.createPrefixChildStream();
  if (encodeTArgs(TParamOut, *Value) == ER_Failure)
    return ER_Failure;

  return encode(Out, Value->getTemplateParam(), true); // {SCOPE NESTING}
}

// -----------------------------------------------------------------------------

// R-L order, reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltDecltypeType> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  // Assembling change (reversed -> ordinary).
  auto DecltypeOut = Out.createPrefixChildStream();
  return encode(DecltypeOut, Value->getDecltype());
}

// -----------------------------------------------------------------------------

// R-L order, scope nesting on inner type, reversed assembling,
// separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltQualType> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  auto QualOut = Out.createPrefixChildStream();
  switch (Value->getKind()) {
  case DTK_Pointer:
    QualOut << " *";
    break;
  case DTK_LValueRef:
    QualOut << " &";
    break;
  case DTK_RValueRef:
    QualOut << " &&";
    break;
  case DTK_C2000Complex:
    QualOut << " _Complex";
    break;
  case DTK_C2000Imaginary:
    QualOut << " _Imaginary";
    break;
  case DTK_PackExpansion:
    QualOut << " ...";
    break;
  default:
    // ReSharper disable once CppUnreachableCode
    assert(false && "Printer does not support qualifier.");
    return ER_Failure;
  }

  return encode(Out, Value->getInnerType()); // {SCOPE NESTING}
}

// -----------------------------------------------------------------------------

// R-L order, scope nesting on inner type, reversed assembling,
// separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltQualGrpType> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  auto QualOut = Out.createPrefixChildStream();
  if (encodeVendorQuals(QualOut, *Value) == ER_Failure)
    return ER_Failure;
  OCLCXX_CLPT_ENCODE(QualOut, Value->getCvrQuals());

  return encode(Out, Value->getInnerType()); // {SCOPE NESTING}
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Nodes (name).

// R-L order, reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out, const std::shared_ptr<const DmngRsltName> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  // {SCOPE NESTING}
  switch (Value->getKind()) {
  case DNK_Ordinary:
    return encode(Out, Value->getAs<DNK_Ordinary>());
  case DNK_Special:
    return encode(Out, Value->getAs<DNK_Special>());
  default:
    // ReSharper disable once CppUnreachableCode
    assert(false && "Printer does not support current node kind.");
    return ER_Failure;
  }
}

// -----------------------------------------------------------------------------

// R-L order on return type / L-R order on name, parameters and qualifiers,
// reversed assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltOrdinaryName> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  auto IsOutPure = Out.isPure();
  auto OrdNameOut = Out.createPrefixChildStream();
  OrdNameOut << (IsOutPure ? " " : " (");

  // {PARENT SCOPE}

  if (encodeNameParts(Out, *Value) == ER_Failure)
    return ER_Failure;
  Out << (IsOutPure ? "" : ")");

  if (encodeSignatureParams(Out, *Value) == ER_Failure)
    return ER_Failure;
  if (encodeVendorQuals(Out, *Value) == ER_Failure)
    return ER_Failure;
  OCLCXX_CLPT_ENCODE(Out, Value->getCvrQuals());
  OCLCXX_CLPT_ENCODE(Out, Value->getRefQuals());

  if (Value->isFunction()) {
    if (encodeSignatureReturn(Out, *Value) == ER_Failure) // {SCOPE NESTING}
      return ER_Failure;
  }

  // {SCOPE PREFIX}
  if (Value->isLocal()) {
    // Assembling change (reversed -> ordinary).
    auto LocalOut = Out.createPrefixChildStream();
    LocalOut << "{";
    // Assembling change (ordinary -> reversed).
    auto ParentOut = LocalOut.createChildStream();
    OCLCXX_CLPT_ENCODE(ParentOut, Value->getLocalScope());
    LocalOut << "}:";
    if (Value->getDefaultValueParamRIdx() >= 0) {
      LocalOut << "param_val``(position: last-"
               << Value->getDefaultValueParamRIdx() << "):";
    }
    else if (Value->isStringLiteral())
      LocalOut << "string``(id: " << Value->getInLocalScopeIdx() << "):";
    else if (Value->getInLocalScopeIdx() > 0)
      LocalOut << "scope``(id: " << Value->getInLocalScopeIdx() << "):";
    LocalOut << " ";
  }

  return ER_Success;
}

// -----------------------------------------------------------------------------

// R-L order, reverse assembling, separate scope required on assembling change.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltSpecialName> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  // Assembling change (reversed -> ordinary).
  auto SpecNameOut = Out.createPrefixChildStream();
  // TODO: Add transactional stubs (GTt) in the future.
  switch (Value->getSpecialKind()) {
  case DSNK_VirtualTable: {
      SpecNameOut << "vtable``{";
      // Assembling change (ordinary -> reversed).
      auto TypeOut = SpecNameOut.createChildStream();
      OCLCXX_CLPT_ENCODE(TypeOut, Value->getRelatedType());
      SpecNameOut << "}";
      break;
    }
  case DSNK_VirtualTableTable: {
      SpecNameOut << "vtt``{";
      // Assembling change (ordinary -> reversed).
      auto TypeOut = SpecNameOut.createChildStream();
      OCLCXX_CLPT_ENCODE(TypeOut, Value->getRelatedType());
      SpecNameOut << "}";
      break;
    }
  case DSNK_TypeInfoStruct: {
      SpecNameOut << "type_info``{";
      // Assembling change (ordinary -> reversed).
      auto TypeOut = SpecNameOut.createChildStream();
      OCLCXX_CLPT_ENCODE(TypeOut, Value->getRelatedType());
      SpecNameOut << "}";
      break;
    }
  case DSNK_TypeInfoNameString: {
      SpecNameOut << "typeid``{";
      // Assembling change (ordinary -> reversed).
      auto TypeOut = SpecNameOut.createChildStream();
      OCLCXX_CLPT_ENCODE(TypeOut, Value->getRelatedType());
      SpecNameOut << "}";
      break;
    }
  case DSNK_VirtualThunk: {
      SpecNameOut << "vthunk``(this: ";
      OCLCXX_CLPT_ENCODE(SpecNameOut, Value->getThisAdjustment());
      SpecNameOut << ", return: ";
      OCLCXX_CLPT_ENCODE(SpecNameOut, Value->getReturnAdjustment());
      SpecNameOut << "){";
      // Assembling change (ordinary -> reversed).
      auto OriginOut = SpecNameOut.createChildStream();
      OCLCXX_CLPT_ENCODE(OriginOut, Value->getOrigin());
      SpecNameOut << "}";
      break;
    }
  case DSNK_GuardVariable: {
      SpecNameOut << "guard``{";
      // Assembling change (ordinary -> reversed).
      auto ObjectOut = SpecNameOut.createChildStream();
      OCLCXX_CLPT_ENCODE(ObjectOut, Value->getRelatedObject());
      SpecNameOut << "}";
      break;
    }
  case DSNK_LifeExtTemporary:  {
      SpecNameOut << "life_ext_tmp``(id: " << Value->getId() << "){";
      // Assembling change (ordinary -> reversed).
      auto ObjectOut = SpecNameOut.createChildStream();
      OCLCXX_CLPT_ENCODE(ObjectOut, Value->getRelatedObject());
      SpecNameOut << "}";
      break;
    }
  default:
    // ReSharper disable once CppUnreachableCode
    assert(false && "Printer does not support current special name kind.");
    return ER_Failure;
  }

  return ER_Success;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Nodes (name parts).

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltNamePart> &Value,
    const std::shared_ptr<const DmngRsltNamePart> &PrevPart) {
  if (Value == nullptr)
    return ER_Failure;

  switch (Value->getPartKind()) {
  case DNPK_Operator:
    return encode(Out, Value->getAs<DNPK_Operator>());
  case DNPK_Constructor:
    return encode(Out, Value->getAs<DNPK_Constructor>(), PrevPart);
  case DNPK_Destructor:
    return encode(Out, Value->getAs<DNPK_Destructor>(), PrevPart);
  case DNPK_Source:
    return encode(Out, Value->getAs<DNPK_Source>());
  case DNPK_UnnamedType:
    return encode(Out, Value->getAs<DNPK_UnnamedType>());
  case DNPK_TemplateParam:
    return encode(Out, Value->getAs<DNPK_TemplateParam>());
  case DNPK_Decltype:
    return encode(Out, Value->getAs<DNPK_Decltype>());
  case DNPK_DataMember:
    return encode(Out, Value->getAs<DNPK_DataMember>());
  default:
    // ReSharper disable once CppUnreachableCode
    assert(false && "Printer does not support current name part kind.");
    return ER_Failure;
  }
}

// -----------------------------------------------------------------------------

EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltOpNamePart> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  if (Value->isConversionOperator()) {
    Out << "operator ";
    // Assembling change (ordinary -> reversed).
    auto ConvTypeOut = Out.createChildStream();
    OCLCXX_CLPT_ENCODE(ConvTypeOut, Value->getConvertTargetType());
    Out << " [[dmng::arity("
        << getInExprOperatorFixedArity(Value->getNameCode())
        << ")]]";
  }
  else if (Value->isLiteralOperator()) {
    Out << "operator \"\" " << Value->getLiteralOperatorSuffix()
        << " [[dmng::arity("
        << getInExprOperatorFixedArity(Value->getNameCode())
        << ")]]";
  }
  else if (Value->isVendorOperator()) {
    Out << "operator " << Value->getVendorOperatorName()
        << " [[dmng::ext_op, dmng::arity(" << Value->getVendorOperatorArity()
        << ")]]";
  }
  else {
    Out << "operator " << getFixedOperatorName(Value->getNameCode())
        << " [[dmng::arity("
        << getInExprOperatorFixedArity(Value->getNameCode())
        << ")]]";
  }
  return encodeTArgs(Out, *Value);
}

// -----------------------------------------------------------------------------

EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltCtorDtorNamePart> &Value,
    const std::shared_ptr<const DmngRsltNamePart> &PrevPart) {
  if (Value == nullptr)
    return ER_Failure;

  StringT FuncName;
  if (PrevPart == nullptr || PrevPart->getPartKind() == DNPK_Source)
    FuncName = PrevPart->getAs<DNPK_Source>()->getSourceName();
  else if (Value->getPartKind() == DNPK_Constructor)
    FuncName = "ctor``";
  else
    FuncName = "dtor``";

  Out << (Value->getPartKind() == DNPK_Constructor ? "" : "~") << FuncName;

  Out << " [[dmng::";
  Out << (Value->getPartKind() == DNPK_Constructor ? "ctor" : "dtor");
  switch (Value->getType()) {
  case DCDT_BaseObj:
    Out << "(base)]]";
    break;
  case DCDT_CompleteObj:
    Out << "(complete)]]";
    break;
  case DCDT_DynMemObj:
    Out << "(alloc)]]";
    break;
  default:
    // ReSharper disable once CppUnreachableCode
    assert(false && "Printer does not support current ctor/dtor type.");
    return ER_Failure;
  }

  return encodeTArgs(Out, *Value);
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltSrcNamePart> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  Out << Value->getSourceName();
  return encodeTArgs(Out, *Value);
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltUnmTypeNamePart> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  Out << (Value->isClosure() ? "lambda``" : "unnamed``")
      << asBase36<StringT>(Value->getId());
  if (encodeSignatureParams(Out, *Value) == ER_Failure)
    return ER_Failure;
  return encodeTArgs(Out, *Value);
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltTParamNamePart> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  OCLCXX_CLPT_ENCODE(Out, Value->getTemplateParam());
  return encodeTArgs(Out, *Value);
}

// -----------------------------------------------------------------------------

// L-R order, ordinary assembling, no scope required.
EncodeResult CxxLikeEncodeTraits::encode(
    OStreamWrapperT &Out,
    const std::shared_ptr<const DmngRsltDecltypeNamePart> &Value) {
  if (Value == nullptr)
    return ER_Failure;

  OCLCXX_CLPT_ENCODE(Out, Value->getDecltype());
  return encodeTArgs(Out, *Value);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------


#undef OCLCXX_CLPT_ENCODE
