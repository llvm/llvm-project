//===- ExtractAPI/Serialization/SerializerBase.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the ExtractAPI APISetVisitor interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SERIALIZERBASE_H
#define LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SERIALIZERBASE_H

#include "clang/ExtractAPI/API.h"

namespace clang {
namespace extractapi {

/// The base interface of visitors for API information.
template <typename Derived> class APISetVisitor {
public:
  void traverseAPISet() {
    getDerived()->traverseGlobalVariableRecords();

    getDerived()->traverseGlobalFunctionRecords();

    getDerived()->traverseEnumRecords();

    getDerived()->traverseStructRecords();

    getDerived()->traverseObjCInterfaces();

    getDerived()->traverseObjCProtocols();

    getDerived()->traverseMacroDefinitionRecords();

    getDerived()->traverseTypedefRecords();
  }

  void traverseGlobalFunctionRecords() {
    for (const auto &GlobalFunction : API.getGlobalFunctions())
      getDerived()->visitGlobalFunctionRecord(*GlobalFunction.second);
  }

  void traverseGlobalVariableRecords() {
    for (const auto &GlobalVariable : API.getGlobalVariables())
      getDerived()->visitGlobalVariableRecord(*GlobalVariable.second);
  }

  void traverseEnumRecords() {
    for (const auto &Enum : API.getEnums())
      getDerived()->visitEnumRecord(*Enum.second);
  }

  void traverseStructRecords() {
    for (const auto &Struct : API.getStructs())
      getDerived()->visitStructRecord(*Struct.second);
  }

  void traverseObjCInterfaces() {
    for (const auto &Interface : API.getObjCInterfaces())
      getDerived()->visitObjCContainerRecord(*Interface.second);
  }

  void traverseObjCProtocols() {
    for (const auto &Protocol : API.getObjCProtocols())
      getDerived()->visitObjCContainerRecord(*Protocol.second);
  }

  void traverseMacroDefinitionRecords() {
    for (const auto &Macro : API.getMacros())
      getDerived()->visitMacroDefinitionRecord(*Macro.second);
  }

  void traverseTypedefRecords() {
    for (const auto &Typedef : API.getTypedefs())
      getDerived()->visitTypedefRecord(*Typedef.second);
  }

  /// Visit a global function record.
  void visitGlobalFunctionRecord(const GlobalFunctionRecord &Record){};

  /// Visit a global variable record.
  void visitGlobalVariableRecord(const GlobalVariableRecord &Record){};

  /// Visit an enum record.
  void visitEnumRecord(const EnumRecord &Record){};

  /// Visit a struct record.
  void visitStructRecord(const StructRecord &Record){};

  /// Visit an Objective-C container record.
  void visitObjCContainerRecord(const ObjCContainerRecord &Record){};

  /// Visit a macro definition record.
  void visitMacroDefinitionRecord(const MacroDefinitionRecord &Record){};

  /// Visit a typedef record.
  void visitTypedefRecord(const TypedefRecord &Record){};

protected:
  const APISet &API;

public:
  APISetVisitor() = delete;
  APISetVisitor(const APISetVisitor &) = delete;
  APISetVisitor(APISetVisitor &&) = delete;
  APISetVisitor &operator=(const APISetVisitor &) = delete;
  APISetVisitor &operator=(APISetVisitor &&) = delete;

protected:
  APISetVisitor(const APISet &API) : API(API) {}
  ~APISetVisitor() = default;

  Derived *getDerived() { return static_cast<Derived *>(this); };
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SERIALIZERBASE_H
