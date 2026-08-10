// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=namespace" %s | FileCheck --check-prefix=CHECK-NAMESPACE %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=type_alias" %s | FileCheck --check-prefix=CHECK-TYPE_ALIAS %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=enum" %s | FileCheck --check-prefix=CHECK-ENUM %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=enum_constant" %s | FileCheck --check-prefix=CHECK-ENUM_CONSTANT %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=record" %s | FileCheck --check-prefix=CHECK-RECORD %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=record(unless(is_union))" %s | FileCheck --check-prefix=CHECK-RECORD_UNLESS_IS_UNION %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=field" %s | FileCheck --check-prefix=CHECK-FIELD %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=function" %s | FileCheck --check-prefix=CHECK-FUNCTION %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=hasType(functionType)" %s | FileCheck --check-prefix=CHECK-HAS_TYPE_FUNCTION_TYPE %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=function(is_member)" %s | FileCheck --check-prefix=CHECK-FUNCTION_IS_MEMBER %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=variable" %s | FileCheck --check-prefix=CHECK-VARIABLE %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=variable(is_global)" %s | FileCheck --check-prefix=CHECK-VARIABLE_IS_GLOBAL %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=variable(is_parameter)" %s | FileCheck --check-prefix=CHECK-VARIABLE_IS_PARAMETER %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter test "-DSUBJECT=variable(unless(is_parameter))" %s | FileCheck --check-prefix=CHECK-VARIABLE_UNLESS_IS_PARAMETER %s

#pragma clang attribute push (__attribute__((annotate("test"))), apply_to = any(SUBJECT))

namespace testNamespace {
// CHECK-NAMESPACE: NamespaceDecl{{.*}} testNamespace
// CHECK-NAMESPACE-NEXT: AnnotateAttr{{.*}} "test"

typedef int testTypedef;
// CHECK-TYPE_ALIAS: TypedefDecl{{.*}} testTypedef
// CHECK-TYPE_ALIAS-NEXT: BuiltinType
// CHECK-TYPE_ALIAS-NEXT: AnnotateAttr{{.*}} "test"

using testTypeAlias = double;
// CHECK-TYPE_ALIAS: TypeAliasDecl{{.*}} testTypeAlias
// CHECK-TYPE_ALIAS-NEXT: BuiltinType
// CHECK-TYPE_ALIAS-NEXT: AnnotateAttr{{.*}} "test"

enum testEnum {
  testEnumCase1,
  testEnumCase2
};
// CHECK-ENUM: EnumDecl{{.*}} testEnum
// CHECK-ENUM-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-ENUM_CONSTANT: EnumConstantDecl{{.*}} testEnumCase1
// CHECK-ENUM_CONSTANT-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-ENUM_CONSTANT: EnumConstantDecl{{.*}} testEnumCase2
// CHECK-ENUM_CONSTANT-NEXT: AnnotateAttr{{.*}} "test"

struct testStructRecord {
  int testStructRecordField;
};
// CHECK-RECORD: CXXRecordDecl{{.*}} testStructRecord
// CHECK-RECORD:   AnnotateAttr{{.*}} "test"
// CHECK-RECORD_UNLESS_IS_UNION-LABEL: CXXRecordDecl{{.*}} testStructRecord
// CHECK-RECORD_UNLESS_IS_UNION:         AnnotateAttr{{.*}} "test"
// CHECK-FIELD: FieldDecl{{.*}} testStructRecordField
// CHECK-FIELD-NEXT: AnnotateAttr{{.*}} "test"

class testClassRecord {
  int testClassRecordField;
};
// CHECK-RECORD: CXXRecordDecl{{.*}} testClassRecord
// CHECK-RECORD:   AnnotateAttr{{.*}} "test"
// CHECK-RECORD_UNLESS_IS_UNION-LABEL: CXXRecordDecl{{.*}} testClassRecord
// CHECK-RECORD_UNLESS_IS_UNION:         AnnotateAttr{{.*}} "test"
// CHECK-FIELD: FieldDecl{{.*}} testClassRecordField
// CHECK-FIELD-NEXT: AnnotateAttr{{.*}} "test"

union testUnionRecord {
  int testUnionRecordField;
};
// CHECK-RECORD: CXXRecordDecl{{.*}} testUnionRecord
// CHECK-RECORD:   AnnotateAttr{{.*}} "test"
// CHECK-RECORD_UNLESS_IS_UNION-LABEL: CXXRecordDecl{{.*}} testUnionRecord
// CHECK-RECORD_UNLESS_IS_UNION-NOT: AnnotateAttr{{.*}} "test"
// CHECK-FIELD: FieldDecl{{.*}} testUnionRecordField
// CHECK-FIELD-NEXT: AnnotateAttr{{.*}} "test"

// CHECK-RECORD_UNLESS_IS_UNION-LABEL: CXXRecordDecl
void testFunctionDecl();
// CHECK-FUNCTION: FunctionDecl{{.*}} testFunctionDecl
// CHECK-FUNCTION-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-HAS_TYPE_FUNCTION_TYPE: FunctionDecl{{.*}} testFunctionDecl
// CHECK-HAS_TYPE_FUNCTION_TYPE-NEXT: AnnotateAttr{{.*}} "test"

void testFunctionDecl() { }
// CHECK-FUNCTION: FunctionDecl{{.*}} testFunctionDecl
// CHECK-FUNCTION-NEXT: CompoundStmt
// CHECK-FUNCTION-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-HAS_TYPE_FUNCTION_TYPE: FunctionDecl{{.*}} testFunctionDecl
// CHECK-HAS_TYPE_FUNCTION_TYPE-NEXT: CompoundStmt
// CHECK-HAS_TYPE_FUNCTION_TYPE-NEXT: AnnotateAttr{{.*}} "test"

void (*testFunctionVar)();
// CHECK-HAS_TYPE_FUNCTION_TYPE: VarDecl{{.*}} testFunctionVar
// CHECK-HAS_TYPE_FUNCTION_TYPE-NEXT: AnnotateAttr{{.*}} "test"
// 'function' should not apply to variables with a function type!
// CHECK-FUNCTION: VarDecl{{.*}} testFunctionVar
// CHECK-FUNCTION-NOT: AnnotateAttr{{.*}} "test"

class testMethods {
  testMethods();
  void testMethod();
};
void testMethods::testMethod() { }
void testFunctionNotMethod();
// CHECK-FUNCTION-LABEL: CXXConstructorDecl{{.*}} testMethods
// CHECK-FUNCTION-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-FUNCTION_IS_MEMBER: CXXConstructorDecl{{.*}} testMethods
// CHECK-FUNCTION_IS_MEMBER-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-HAS_TYPE_FUNCTION_TYPE: CXXConstructorDecl{{.*}} testMethods
// CHECK-HAS_TYPE_FUNCTION_TYPE-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-FUNCTION: CXXMethodDecl{{.*}} testMethod
// CHECK-FUNCTION-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-FUNCTION_IS_MEMBER: CXXMethodDecl{{.*}} testMethod
// CHECK-FUNCTION_IS_MEMBER-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-HAS_TYPE_FUNCTION_TYPE: CXXMethodDecl{{.*}} testMethod
// CHECK-HAS_TYPE_FUNCTION_TYPE-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-FUNCTION: CXXMethodDecl{{.*}} testMethod
// CHECK-FUNCTION-NEXT: CompoundStmt
// CHECK-FUNCTION-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-FUNCTION_IS_MEMBER: CXXMethodDecl{{.*}} testMethod
// CHECK-FUNCTION_IS_MEMBER-NEXT: CompoundStmt
// CHECK-CXX_METHOD-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-HAS_TYPE_FUNCTION_TYPE: CXXMethodDecl{{.*}} testMethod
// CHECK-HAS_TYPE_FUNCTION_TYPE-NEXT: CompoundStmt
// CHECK-HAS_TYPE_FUNCTION_TYPE-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-FUNCTION_IS_MEMBER: FunctionDecl{{.*}} testFunctionNotMethod
// CHECK-FUNCTION_IS_MEMBER-NOT: AnnotateAttr{{.*}} "test"

int testVariable;
// CHECK-VARIABLE: VarDecl{{.*}} testVariable
// CHECK-VARIABLE-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-VARIABLE_IS_GLOBAL-LABEL: VarDecl{{.*}} testVariable
// CHECK-VARIABLE_IS_GLOBAL-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-VARIABLE_IS_PARAMETER-LABEL: VarDecl{{.*}} testVariable
// CHECK-VARIABLE_IS_PARAMETER-NOT: AnnotateAttr{{.*}} "test"
// CHECK-VARIABLE_UNLESS_IS_PARAMETER-LABEL: VarDecl{{.*}} testVariable
// CHECK-VARIABLE_UNLESS_IS_PARAMETER-NEXT: AnnotateAttr{{.*}} "test"
void testVarFunction(int testParam) {
// CHECK-VARIABLE: VarDecl{{.*}} testParam
// CHECK-VARIABLE-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-VARIABLE_IS_GLOBAL-LABEL: VarDecl{{.*}} testParam
// CHECK-VARIABLE_IS_GLOBAL-NOT: AnnotateAttr{{.*}} "test"
// CHECK-VARIABLE_IS_PARAMETER-LABEL: VarDecl{{.*}} testParam
// CHECK-VARIABLE_IS_PARAMETER-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-VARIABLE_UNLESS_IS_PARAMETER-LABEL: VarDecl{{.*}} testParam
// CHECK-VARIABLE_UNLESS_IS_PARAMETER-NOT: AnnotateAttr{{.*}} "test"

  int testLocalVariable;
// CHECK-VARIABLE: VarDecl{{.*}} testLocalVariable
// CHECK-VARIABLE-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-VARIABLE_IS_GLOBAL-LABEL: VarDecl{{.*}} testLocalVariable
// CHECK-VARIABLE_IS_GLOBAL-NOT: AnnotateAttr{{.*}} "test"
// CHECK-VARIABLE_IS_PARAMETER-LABEL: VarDecl{{.*}} testLocalVariable
// CHECK-VARIABLE_IS_PARAMETER-NOT: AnnotateAttr{{.*}} "test"
// CHECK-VARIABLE_UNLESS_IS_PARAMETER-LABEL: VarDecl{{.*}} testLocalVariable
// CHECK-VARIABLE_UNLESS_IS_PARAMETER-NEXT: AnnotateAttr{{.*}} "test"
}
class testVarClass {
  static int testStaticVar;
};
// CHECK-VARIABLE: VarDecl{{.*}} testStaticVar
// CHECK-VARIABLE-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-VARIABLE_IS_GLOBAL-LABEL: VarDecl{{.*}} testStaticVar
// CHECK-VARIABLE_IS_GLOBAL-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-VARIABLE_IS_PARAMETER-LABEL: VarDecl{{.*}} testStaticVar
// CHECK-VARIABLE_IS_PARAMETER-NOT: AnnotateAttr{{.*}} "test"
// CHECK-VARIABLE_UNLESS_IS_PARAMETER-LABEL: VarDecl{{.*}} testStaticVar
// CHECK-VARIABLE_UNLESS_IS_PARAMETER-NEXT: AnnotateAttr{{.*}} "test"


}

#pragma clang attribute pop
