//===--- ValuePrinter.cpp - Value Printer -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements some value printer functions for clang-repl.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Interpreter/Value.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ParsedAttr.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include "InterpreterUtils.h"

#include <cassert>
#include <codecvt>
#include <cstddef>
#include <cstdint>
#include <locale>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace clang {

std::string printAddress(const void *Ptr, const char Prefix = 0) {
  if (!Ptr) {
    return "nullptr";
  }
  std::ostringstream ostr;
  if (Prefix) {
    ostr << Prefix;
  }
  ostr << Ptr;
  return ostr.str();
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const void *ptr) { return printAddress(ptr, '@'); }

REPL_EXTERNAL_VISIBILITY
std::string printValue(const void **ptr) { return printAddress(*ptr); }

REPL_EXTERNAL_VISIBILITY
std::string printValue(const bool *ptr) { return *ptr ? "true" : "false"; }

REPL_EXTERNAL_VISIBILITY
std::string printValue(const char *ptr) {
  std::string value = "'";
  switch (*ptr) {
  case '\t':
    value += "\\t";
    break;
  case '\n':
    value += "\\n";
    break;
  case '\r':
    value += "\\r";
    break;
  case '\f':
    value += "\\f";
    break;
  case '\v':
    value += "\\v";
    break;
  default:
    value += *ptr;
  }
  value += "'";
  return value;
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const signed char *ptr) {
  std::string value = "'";
  switch (*ptr) {
  case '\t':
    value += "\\t";
    break;
  case '\n':
    value += "\\n";
    break;
  case '\r':
    value += "\\r";
    break;
  case '\f':
    value += "\\f";
    break;
  case '\v':
    value += "\\v";
    break;
  default:
    value += *ptr;
  }
  value += "'";
  return value;
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const unsigned char *ptr) {
  std::string value = "'";
  switch (*ptr) {
  case '\t':
    value += "\\t";
    break;
  case '\n':
    value += "\\n";
    break;
  case '\r':
    value += "\\r";
    break;
  case '\f':
    value += "\\f";
    break;
  case '\v':
    value += "\\v";
    break;
  default:
    value += *ptr;
  }
  value += "'";
  return value;
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const wchar_t *ptr) {
  std::string value = "'";
  switch (*ptr) {
  case '\t':
    value += "\\t";
    break;
  case '\n':
    value += "\\n";
    break;
  case '\r':
    value += "\\r";
    break;
  case '\f':
    value += "\\f";
    break;
  case '\v':
    value += "\\v";
    break;
  default:
    value += *ptr;
  }
  value += "'";
  return value;
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const char16_t *ptr) {
  std::string value = "'";
  switch (*ptr) {
  case '\t':
    value += "\\t";
    break;
  case '\n':
    value += "\\n";
    break;
  case '\r':
    value += "\\r";
    break;
  case '\f':
    value += "\\f";
    break;
  case '\v':
    value += "\\v";
    break;
  default:
    value += *ptr;
  }
  value += "'";
  return value;
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const char32_t *ptr) {
  std::string value = "'";
  switch (*ptr) {
  case '\t':
    value += "\\t";
    break;
  case '\n':
    value += "\\n";
    break;
  case '\r':
    value += "\\r";
    break;
  case '\f':
    value += "\\f";
    break;
  case '\v':
    value += "\\v";
    break;
  default:
    value += *ptr;
  }
  value += "'";
  return value;
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const unsigned short *ptr) {
  std::ostringstream ostr;
  ostr << *ptr;
  return ostr.str();
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const short *ptr) {
  std::ostringstream ostr;
  ostr << *ptr;
  return ostr.str();
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const unsigned int *ptr) {
  std::ostringstream ostr;
  ostr << *ptr;
  return ostr.str();
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const int *ptr) {
  std::ostringstream ostr;
  ostr << *ptr;
  return ostr.str();
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const unsigned long *ptr) {
  std::ostringstream ostr;
  ostr << *ptr;
  return ostr.str();
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const long *ptr) {
  std::ostringstream ostr;
  ostr << *ptr;
  return ostr.str();
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const unsigned long long *ptr) {
  std::ostringstream ostr;
  ostr << *ptr;
  return ostr.str();
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const long long *ptr) {
  std::ostringstream ostr;
  ostr << *ptr;
  return ostr.str();
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const float *ptr) {
  std::ostringstream ostr;
  ostr << *ptr;
  return ostr.str();
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const double *ptr) {
  std::ostringstream ostr;
  ostr << *ptr;
  return ostr.str();
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const long double *ptr) {
  std::ostringstream ostr;
  ostr << *ptr;
  return ostr.str();
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(char *ptr, bool seq) {
  std::string value = "\"";
  char *p = ptr;
  while (*p != '\0') {
    value += *p;
    p++;
  }
  value += "\"";
  return value;
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(signed char *ptr, bool seq) {
  std::string value = "\"";
  signed char *p = ptr;
  while (*p != '\0') {
    value += *p;
    p++;
  }
  value += "\"";
  return value;
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(unsigned char *ptr, bool seq) {
  std::string value = "\"";
  unsigned char *p = ptr;
  while (*p != '\0') {
    value += *p;
    p++;
  }
  value += "\"";
  return value;
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(wchar_t *ptr, bool seq) {
  std::wstring value;
  wchar_t *p = ptr;
  while (*p != '\0') {
    value += *p;
    p++;
  }
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  return "\"" + converter.to_bytes(value) + "\"";
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(char16_t *ptr, bool seq) {
  std::u16string value;
  char16_t *p = ptr;
  while (*p != '\0') {
    value += *p;
    p++;
  }
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> converter;
  return "\"" + converter.to_bytes(value) + "\"";
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(char32_t *ptr, bool seq) {
  std::u32string value;
  char32_t *p = ptr;
  while (*p != '\0') {
    value += *p;
    p++;
  }
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
  return "\"" + converter.to_bytes(value) + "\"";
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const std::string *ptr) { return "\"" + *ptr + "\""; }

REPL_EXTERNAL_VISIBILITY
std::string printValue(const std::wstring *ptr) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  return "\"" + converter.to_bytes(*ptr) + "\"";
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const std::u16string *ptr) {
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> converter;
  return "\"" + converter.to_bytes(*ptr) + "\"";
}

REPL_EXTERNAL_VISIBILITY
std::string printValue(const std::u32string *ptr) {
  std::wstring_convert<std::codecvt_utf8_utf16<char32_t>, char32_t> converter;
  return "\"" + converter.to_bytes(*ptr) + "\"";
}

std::string printBuiltinTypeValue(const Value &V, BuiltinType::Kind Kind) {
  switch (Kind) {
  case BuiltinType::Bool:
    return V.convertTo<bool>() ? "true" : "false";
  case BuiltinType::Char_U:
  case BuiltinType::UChar: {
    auto val = V.convertTo<unsigned char>();
    return printValue(&val);
  }
  case BuiltinType::Char_S:
  case BuiltinType::SChar: {
    auto val = V.convertTo<signed char>();
    return printValue(&val);
  }
  case BuiltinType::WChar_S: {
    auto val = V.convertTo<wchar_t>();
    return printValue(&val);
  }
  case BuiltinType::Char16: {
    auto val = V.convertTo<char16_t>();
    return printValue(&val);
  }
  case BuiltinType::Char32: {
    auto val = V.convertTo<char32_t>();
    return printValue(&val);
  }
  case BuiltinType::UShort:
    return std::to_string(V.convertTo<unsigned short>());
  case BuiltinType::Short:
    return std::to_string(V.convertTo<short>());
  case BuiltinType::UInt:
    return std::to_string(V.convertTo<unsigned int>());
  case BuiltinType::Int:
    return std::to_string(V.convertTo<int>());
  case BuiltinType::ULong:
    return std::to_string(V.convertTo<unsigned long>());
  case BuiltinType::Long:
    return std::to_string(V.convertTo<long>());
  case BuiltinType::ULongLong:
    return std::to_string(V.convertTo<unsigned long long>());
  case BuiltinType::LongLong:
    return std::to_string(V.convertTo<long long>());
  case BuiltinType::Float:
    return std::to_string(V.convertTo<float>());
  case BuiltinType::Double:
    return std::to_string(V.convertTo<double>());
  case BuiltinType::LongDouble:
    return std::to_string(V.convertTo<long double>());
  default:
    break;
  }
  return "";
}

std::string printValueByPtr(void *Ptr, QualType Type, uint64_t Offset) {
  if (const BuiltinType *bt =
          llvm::dyn_cast<BuiltinType>(Type.getCanonicalType().getTypePtr())) {
    std::ostringstream os;
    Ptr = (void *)((char *)Ptr + Offset);
    switch (bt->getKind()) {
    case BuiltinType::Bool:
      return printValue((bool *)Ptr);
    case BuiltinType::Char_U:
      return printValue((unsigned char *)Ptr);
    case BuiltinType::UChar:
      return printValue((unsigned char *)Ptr);
    case BuiltinType::Char_S:
      return printValue((signed char *)Ptr);
    case BuiltinType::SChar:
      return printValue((signed char *)Ptr);
    case BuiltinType::WChar_S:
      return printValue((wchar_t *)Ptr);
    case BuiltinType::Char16:
      return printValue((char16_t *)Ptr);
    case BuiltinType::Char32:
      return printValue((char32_t *)Ptr);
    case BuiltinType::UShort:
      return printValue((unsigned short *)Ptr);
    case BuiltinType::Short:
      return printValue((short *)Ptr);
    case BuiltinType::UInt:
      return printValue((unsigned int *)Ptr);
    case BuiltinType::Int:
      return printValue((int *)Ptr);
    case BuiltinType::ULong:
      return printValue((unsigned long *)Ptr);
    case BuiltinType::Long:
      return printValue((long *)Ptr);
    case BuiltinType::ULongLong:
      return printValue((unsigned long long *)Ptr);
    case BuiltinType::LongLong:
      return printValue((long long *)Ptr);
    case BuiltinType::Float:
      return printValue((float *)Ptr);
    case BuiltinType::Double:
      return printValue((double *)Ptr);
    case BuiltinType::LongDouble:
      return printValue((long double *)Ptr);
    default:
      break;
    }
  }
  if (!Ptr) {
    return "nullptr";
  }
  return printAddress(Ptr, '@');
}

std::string printEnumValue(const Value &V, QualType Type) {
  std::ostringstream ostr;
  const ASTContext &C = V.getASTContext();
  const EnumType *EnumTy = Type->getAs<EnumType>();
  assert(EnumTy && "printEnumValue invoked for a non enum type");
  EnumDecl *decl = EnumTy->getDecl();
  uint64_t value = V.getULongLong();
  bool isFirst = true;
  llvm::APSInt valAsAPSInt = C.MakeIntValue(value, Type);
  for (EnumDecl::enumerator_iterator I = decl->enumerator_begin(),
                                     E = decl->enumerator_end();
       I != E; ++I) {
    if (I->getInitVal() == valAsAPSInt) {
      if (!isFirst) {
        ostr << " ? ";
      }
      ostr << "(" << I->getQualifiedNameAsString() << ")";
      isFirst = false;
    }
  }
  ostr << " : " << decl->getIntegerType().getAsString() << " "
       << llvm::toString(valAsAPSInt, 10);
  return ostr.str();
}

std::string printArrayValue(const Value &V, QualType Type) {
  if (const ArrayType *decl = Type->getAsArrayTypeUnsafe()) {
    if (const ConstantArrayType *cdecl =
            dyn_cast<const ConstantArrayType>(decl)) {
      QualType elemType = decl->getElementType();
      uint64_t size = cdecl->getSize().getZExtValue();
      std::ostringstream ostr;
      ostr << "{ ";
      const ASTContext &C = V.getASTContext();
      uint64_t elemSize = C.getTypeSize(elemType) / 8;
      for (uint64_t i = 0; i < size; i++) {
        ostr << printValueByPtr(V.getPtr(), elemType, i * elemSize);
        if (i != size - 1) {
          ostr << ", ";
        }
      }
      ostr << " }";
      return ostr.str();
    }
  }

  if (Type->isPointerType()) {
    QualType elemType = Type->getPointeeType();
    if (elemType->isPointerType()) {
      return "";
    }
    const BuiltinType *bt = llvm::dyn_cast<BuiltinType>(elemType.getTypePtr());
    BuiltinType::Kind Kind = bt->getKind();
    switch (Kind) {
    case BuiltinType::Char_U:
      return printValue((unsigned char *)V.getPtr(), true);
    case BuiltinType::UChar:
      return printValue((unsigned char *)V.getPtr(), true);
    case BuiltinType::Char_S:
      return printValue((signed char *)V.getPtr(), true);
    case BuiltinType::SChar:
      return printValue((signed char *)V.getPtr(), true);
    case BuiltinType::WChar_S:
      return printValue((wchar_t *)V.getPtr(), true);
    case BuiltinType::Char16:
      return printValue((char16_t *)V.getPtr(), true);
    case BuiltinType::Char32:
      return printValue((char32_t *)V.getPtr(), true);
    default:
      break;
    }
  }
  return "";
}

std::string printStringValue(const Value &V, QualType Type) {
  const ASTContext &C = V.getASTContext();
  std::string typeStr = GetFullTypeName(C, Type);
  std::string stringVal = "";
  if (typeStr == "std::string")
    stringVal = printValue((std::string *)V.getPtr());
  if (typeStr == "std::wstring")
    stringVal = printValue((std::wstring *)V.getPtr());
  if (typeStr == "std::u16string")
    stringVal = printValue((std::u16string *)V.getPtr());
  if (typeStr == "std::u32string")
    stringVal = printValue((std::u32string *)V.getPtr());
  return stringVal;
}

template <typename T> std::string printVectorValue(void *ptr) {
  std::vector<T> *vecPtr = static_cast<std::vector<T> *>(ptr);
  std::ostringstream ostr;
  ostr << "{ ";
  for (uint64_t i = 0; i < vecPtr->size(); i++) {
    ostr << (*vecPtr)[i];
    if (i != vecPtr->size() - 1) {
      ostr << ", ";
    }
  }
  ostr << " }";
  return ostr.str();
}

std::string printRecordValue(const Value &V, const CXXRecordDecl *RecordDecl) {
  const ASTContext &C = V.getASTContext();
  Sema &S = V.getInterpreter().getSema();
  std::string Name = V.getName();
  NamedDecl *namedDecl = LookupNamed(S, Name, nullptr);

  ValueDecl *valueDecl = llvm::dyn_cast<ValueDecl>(namedDecl);
  if (valueDecl != nullptr) {
    QualType Type = valueDecl->getType();
    std::string str = printStringValue(V, Type);
    if (!str.empty()) {
      return str;
    }

    if (llvm::StringRef(GetFullTypeName(C, Type)).starts_with("std::vector")) {
      const auto *specDecl =
          dyn_cast<ClassTemplateSpecializationDecl>(RecordDecl);
      if (specDecl) {
        const TemplateArgumentList &tplArgs = specDecl->getTemplateArgs();
        assert(tplArgs.size() != 0);
        const TemplateArgument &tplArg = tplArgs[0];
        if (tplArg.getKind() == TemplateArgument::Type) {
          if (const BuiltinType *BT =
                  dyn_cast<BuiltinType>(tplArg.getAsType().getTypePtr())) {
            void *ptr = V.getPtr();
            switch (BT->getKind()) {
            case BuiltinType::Bool:
              return printVectorValue<bool>(ptr);
            case BuiltinType::Char_U:
            case BuiltinType::UChar:
              return printVectorValue<unsigned char>(ptr);
            case BuiltinType::Char_S:
            case BuiltinType::SChar:
              return printVectorValue<signed char>(ptr);
            case BuiltinType::WChar_S:
              return printVectorValue<wchar_t>(ptr);
            case BuiltinType::Char16:
              return printVectorValue<char16_t>(ptr);
            case BuiltinType::Char32:
              return printVectorValue<char32_t>(ptr);
            case BuiltinType::UShort:
              return printVectorValue<unsigned short>(ptr);
            case BuiltinType::Short:
              return printVectorValue<short>(ptr);
            case BuiltinType::UInt:
              return printVectorValue<unsigned int>(ptr);
            case BuiltinType::Int:
              return printVectorValue<int>(ptr);
            case BuiltinType::ULong:
              return printVectorValue<unsigned long>(ptr);
            case BuiltinType::Long:
              return printVectorValue<long>(ptr);
            case BuiltinType::ULongLong:
              return printVectorValue<unsigned long long>(ptr);
            case BuiltinType::LongLong:
              return printVectorValue<long long>(ptr);
            case BuiltinType::Float:
              return printVectorValue<float>(ptr);
            case BuiltinType::Double:
              return printVectorValue<double>(ptr);
            case BuiltinType::LongDouble:
              return printVectorValue<long double>(ptr);
            default:
              break;
            }
          }
        }
      }

      return "";
    }
  }

  std::ostringstream values;
  auto fields = RecordDecl->fields();
  for (auto I = fields.begin(), E = fields.end(), tmp = I; I != E;
       ++I, tmp = I) {
    FieldDecl *field = *I;
    QualType fieldType = field->getType();
    size_t offset = C.getFieldOffset(field) /* bits */ / 8;
    values << field->getNameAsString() << ": "
           << printValueByPtr(V.getPtr(), fieldType, offset);
    if (++tmp != E) {
      values << ", ";
    }
  }
  if (!values.str().empty()) {
    return "{ " + values.str() + " }";
  }
  return "";
}

std::string printUnpackedValue(const Value &V) {
  const ASTContext &C = V.getASTContext();
  const QualType Td = V.getType().getDesugaredType(C);
  const QualType Ty = Td.getNonReferenceType();

  if (!V.getPtr()) {
    return "nullptr";
  }
  if (Ty->isNullPtrType()) {
    return "nullptr_t";
  }
  if (Ty->isEnumeralType()) {
    return printEnumValue(V, Ty);
  }
  if (CXXRecordDecl *RecordDecl = Ty->getAsCXXRecordDecl()) {
    if (RecordDecl->isLambda()) {
      return printAddress(V.getPtr(), '@');
    }
    std::string str = printRecordValue(V, RecordDecl);
    if (!str.empty()) {
      return str;
    }
  } else if (const BuiltinType *BT = llvm::dyn_cast<BuiltinType>(
                 Td.getCanonicalType().getTypePtr())) {
    BuiltinType::Kind Kind = BT->getKind();
    std::string str = printBuiltinTypeValue(V, Kind);
    if (!str.empty()) {
      return str;
    }
  } else {
    std::string str = printArrayValue(V, Ty);
    if (!str.empty()) {
      return str;
    }
  }
  return printAddress(V.getPtr(), '@');
}

std::string Value::printValueInternal() const {
  return printUnpackedValue(*this);
}

} // namespace clang
