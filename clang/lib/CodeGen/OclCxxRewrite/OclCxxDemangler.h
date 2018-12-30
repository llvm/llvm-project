//===- OclCxxDemangler.h - OCLC++ simple demangler              -*- C++ -*-===//
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


#ifndef CLANG_LIB_CODEGEN_OCLCXXREWRITE_OCLCXXDEMANGLER_H
#define CLANG_LIB_CODEGEN_OCLCXXREWRITE_OCLCXXDEMANGLER_H

#include "OclCxxDemanglerResult.h"

#include <string>


namespace oclcxx {
namespace adaptation {

/// \brief Returns C++ name of built-in type.
///
/// Handles all built-in types except vendor-extended types.
const std::string &getFixedBuiltinTypeName(DmngBuiltinType Type);

/// \brief Returns Itanium-encoded name of built-in type.
///
/// Handles all built-in types except vendor-extended types.
const std::string &getEncFixedBuiltinTypeName(DmngBuiltinType Type);

/// \brief Returns C++ name of operator  (without "operator" prefix).
///
/// Handles all operator names except convert, literal and vendor-extended
/// operators.
const std::string &getFixedOperatorName(DmngOperatorName NameCode);

/// \brief Returns Itanium-encoded name of operator.
///
/// Handles all operator names except convert, literal and vendor-extended
/// operators.
const std::string &getEncFixedOperatorName(DmngOperatorName NameCode);

/// \brief Returns fixed arity of operator in <expression> context.
///
/// \return Arity of operator, or 0 if arity is unknown, operator requires
///         special form in <expression> or number of operands is variable.
int getInExprOperatorFixedArity(DmngOperatorName NameCode);


/// \brief Parser for Itanium-mangled names.
class ItaniumNameParser {
public:
  /// Function type which can return address space based on vendor-extended
  /// qualifier.
  using AddressSpaceExtractFuncT =
    DmngAddressSpaceQuals (const DmngRsltVendorQual &);


  /// \brief Parses mangled name (by copy).
  DmngRslt parse(const std::string &MangledName) {
    return parse(std::string(MangledName));
  }

  /// \brief Parses mangled name (by move).
  DmngRslt parse(std::string &&MangledName);


  /// \brief Creates new instance of parser for Itanium-mangled names.
  ///
  /// \param ASExtractFunc Address-space extract function. Function return
  ///                      address-space, or DASQ_None if vendor-extended
  ///                      qualifier does not contain address space.
  explicit ItaniumNameParser(AddressSpaceExtractFuncT &ASExtractFunc)
    : ASExtractFunc(ASExtractFunc) {}

private:
  /// Address-space extract function.
  AddressSpaceExtractFuncT &ASExtractFunc;
};


/// \brief Address space extract function for OpenCL C++.
inline DmngAddressSpaceQuals OclASExtract(const DmngRsltVendorQual &Qual) {
  if (Qual.getName().size() < 3)
    return DASQ_None;

  switch (Qual.getName()[0]) {
  case 'A':
    if (Qual.getName()[1] == 'S' && Qual.getName().size() == 3) {
      switch (Qual.getName()[2]) {
      case '0': return DASQ_Private;
      case '3': return DASQ_Local;
      case '1': return DASQ_Global;
      case '2': return DASQ_Constant;
      case '4': return DASQ_Generic;
      default: return DASQ_None;
      }
    }
    break;
  case 'C':
    if (Qual.getName() == "CLlocal")
      return DASQ_Local;
    if (Qual.getName() == "CLglobal")
      return DASQ_Global;
    if (Qual.getName() == "CLconstant")
      return DASQ_Constant;
    break;
  }
  return DASQ_None;
}

} // adaptation
} // oclcxx

#endif // CLANG_LIB_CODEGEN_OCLCXXREWRITE_OCLCXXDEMANGLER_H
