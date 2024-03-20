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
    getDerived()->traverseNamespaces();

    getDerived()->traverseGlobalVariableRecords();

    getDerived()->traverseGlobalFunctionRecords();

    getDerived()->traverseEnumRecords();

    getDerived()->traverseStaticFieldRecords();

    getDerived()->traverseCXXClassRecords();

    getDerived()->traverseClassTemplateRecords();

    getDerived()->traverseClassTemplateSpecializationRecords();

    getDerived()->traverseClassTemplatePartialSpecializationRecords();

    getDerived()->traverseCXXInstanceMethods();

    getDerived()->traverseCXXStaticMethods();

    getDerived()->traverseCXXMethodTemplates();

    getDerived()->traverseCXXMethodTemplateSpecializations();

    getDerived()->traverseCXXFields();

    getDerived()->traverseCXXFieldTemplates();

    getDerived()->traverseConcepts();

    getDerived()->traverseGlobalVariableTemplateRecords();

    getDerived()->traverseGlobalVariableTemplateSpecializationRecords();

    getDerived()->traverseGlobalVariableTemplatePartialSpecializationRecords();

    getDerived()->traverseGlobalFunctionTemplateRecords();

    getDerived()->traverseGlobalFunctionTemplateSpecializationRecords();

    getDerived()->traverseRecordRecords();

    getDerived()->traverseObjCInterfaces();

    getDerived()->traverseObjCProtocols();

    getDerived()->traverseObjCCategories();

    getDerived()->traverseMacroDefinitionRecords();

    getDerived()->traverseTypedefRecords();
  }

  void traverseNamespaces() {
    for (const auto &Namespace : API.getNamespaces())
      getDerived()->visitNamespaceRecord(*Namespace.second);
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

  void traverseRecordRecords() {
    for (const auto &Record : API.getRecords())
      getDerived()->visitRecordRecord(*Record.second);
  }

  void traverseStaticFieldRecords() {
    for (const auto &StaticField : API.getStaticFields())
      getDerived()->visitStaticFieldRecord(*StaticField.second);
  }

  void traverseCXXClassRecords() {
    for (const auto &Class : API.getCXXClasses())
      getDerived()->visitCXXClassRecord(*Class.second);
  }

  void traverseCXXMethodTemplates() {
    for (const auto &MethodTemplate : API.getCXXMethodTemplates())
      getDerived()->visitMethodTemplateRecord(*MethodTemplate.second);
  }

  void traverseCXXMethodTemplateSpecializations() {
    for (const auto &MethodTemplateSpecialization :
         API.getCXXMethodTemplateSpecializations())
      getDerived()->visitMethodTemplateSpecializationRecord(
          *MethodTemplateSpecialization.second);
  }

  void traverseClassTemplateRecords() {
    for (const auto &ClassTemplate : API.getClassTemplates())
      getDerived()->visitClassTemplateRecord(*ClassTemplate.second);
  }

  void traverseClassTemplateSpecializationRecords() {
    for (const auto &ClassTemplateSpecialization :
         API.getClassTemplateSpecializations())
      getDerived()->visitClassTemplateSpecializationRecord(
          *ClassTemplateSpecialization.second);
  }

  void traverseClassTemplatePartialSpecializationRecords() {
    for (const auto &ClassTemplatePartialSpecialization :
         API.getClassTemplatePartialSpecializations())
      getDerived()->visitClassTemplatePartialSpecializationRecord(
          *ClassTemplatePartialSpecialization.second);
  }

  void traverseCXXInstanceMethods() {
    for (const auto &InstanceMethod : API.getCXXInstanceMethods())
      getDerived()->visitCXXInstanceMethodRecord(*InstanceMethod.second);
  }

  void traverseCXXStaticMethods() {
    for (const auto &InstanceMethod : API.getCXXStaticMethods())
      getDerived()->visitCXXStaticMethodRecord(*InstanceMethod.second);
  }

  void traverseCXXFields() {
    for (const auto &CXXField : API.getCXXFields())
      getDerived()->visitCXXFieldRecord(*CXXField.second);
  }

  void traverseCXXFieldTemplates() {
    for (const auto &CXXFieldTemplate : API.getCXXFieldTemplates())
      getDerived()->visitCXXFieldTemplateRecord(*CXXFieldTemplate.second);
  }

  void traverseGlobalVariableTemplateRecords() {
    for (const auto &GlobalVariableTemplate : API.getGlobalVariableTemplates())
      getDerived()->visitGlobalVariableTemplateRecord(
          *GlobalVariableTemplate.second);
  }

  void traverseGlobalVariableTemplateSpecializationRecords() {
    for (const auto &GlobalVariableTemplateSpecialization :
         API.getGlobalVariableTemplateSpecializations())
      getDerived()->visitGlobalVariableTemplateSpecializationRecord(
          *GlobalVariableTemplateSpecialization.second);
  }

  void traverseGlobalVariableTemplatePartialSpecializationRecords() {
    for (const auto &GlobalVariableTemplatePartialSpecialization :
         API.getGlobalVariableTemplatePartialSpecializations())
      getDerived()->visitGlobalVariableTemplatePartialSpecializationRecord(
          *GlobalVariableTemplatePartialSpecialization.second);
  }

  void traverseGlobalFunctionTemplateRecords() {
    for (const auto &GlobalFunctionTemplate : API.getGlobalFunctionTemplates())
      getDerived()->visitGlobalFunctionTemplateRecord(
          *GlobalFunctionTemplate.second);
  }

  void traverseGlobalFunctionTemplateSpecializationRecords() {
    for (const auto &GlobalFunctionTemplateSpecialization :
         API.getGlobalFunctionTemplateSpecializations())
      getDerived()->visitGlobalFunctionTemplateSpecializationRecord(
          *GlobalFunctionTemplateSpecialization.second);
  }

  void traverseConcepts() {
    for (const auto &Concept : API.getConcepts())
      getDerived()->visitConceptRecord(*Concept.second);
  }

  void traverseObjCInterfaces() {
    for (const auto &Interface : API.getObjCInterfaces())
      getDerived()->visitObjCContainerRecord(*Interface.second);
  }

  void traverseObjCProtocols() {
    for (const auto &Protocol : API.getObjCProtocols())
      getDerived()->visitObjCContainerRecord(*Protocol.second);
  }

  void traverseObjCCategories() {
    for (const auto &Category : API.getObjCCategories())
      getDerived()->visitObjCCategoryRecord(*Category.second);
  }

  void traverseMacroDefinitionRecords() {
    for (const auto &Macro : API.getMacros())
      getDerived()->visitMacroDefinitionRecord(*Macro.second);
  }

  void traverseTypedefRecords() {
    for (const auto &Typedef : API.getTypedefs())
      getDerived()->visitTypedefRecord(*Typedef.second);
  }

  void visitNamespaceRecord(const NamespaceRecord &Record){};

  /// Visit a global function record.
  void visitGlobalFunctionRecord(const GlobalFunctionRecord &Record){};

  /// Visit a global variable record.
  void visitGlobalVariableRecord(const GlobalVariableRecord &Record){};

  /// Visit an enum record.
  void visitEnumRecord(const EnumRecord &Record){};

  /// Visit a record record.
  void visitRecordRecord(const RecordRecord &Record){};

  void visitStaticFieldRecord(const StaticFieldRecord &Record){};

  void visitCXXClassRecord(const CXXClassRecord &Record){};

  void visitClassTemplateRecord(const ClassTemplateRecord &Record){};

  void visitClassTemplateSpecializationRecord(
      const ClassTemplateSpecializationRecord &Record){};

  void visitClassTemplatePartialSpecializationRecord(
      const ClassTemplatePartialSpecializationRecord &Record){};

  void visitCXXInstanceRecord(const CXXInstanceMethodRecord &Record){};

  void visitCXXStaticRecord(const CXXStaticMethodRecord &Record){};

  void visitMethodTemplateRecord(const CXXMethodTemplateRecord &Record){};

  void visitMethodTemplateSpecializationRecord(
      const CXXMethodTemplateSpecializationRecord &Record){};

  void visitCXXFieldTemplateRecord(const CXXFieldTemplateRecord &Record){};

  void visitGlobalVariableTemplateRecord(
      const GlobalVariableTemplateRecord &Record) {}

  void visitGlobalVariableTemplateSpecializationRecord(
      const GlobalVariableTemplateSpecializationRecord &Record){};

  void visitGlobalVariableTemplatePartialSpecializationRecord(
      const GlobalVariableTemplatePartialSpecializationRecord &Record){};

  void visitGlobalFunctionTemplateRecord(
      const GlobalFunctionTemplateRecord &Record){};

  void visitGlobalFunctionTemplateSpecializationRecord(
      const GlobalFunctionTemplateSpecializationRecord &Record){};

  /// Visit an Objective-C container record.
  void visitObjCContainerRecord(const ObjCContainerRecord &Record){};

  /// Visit an Objective-C category record.
  void visitObjCCategoryRecord(const ObjCCategoryRecord &Record){};

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
