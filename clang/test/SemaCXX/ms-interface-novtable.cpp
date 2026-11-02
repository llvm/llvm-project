// RUN: %clang_cc1 -triple x86_64-pc-win32-msvc %s -fsyntax-only -ast-dump -fms-extensions -Wno-microsoft -std=c++11 | FileCheck %s

__interface I1 {
};

// Use of __interface implies __declspec(novtable); see GH232046.
// CHECK: CXXRecordDecl {{.*}} __interface I1 definition
// CHECK: MSNoVTableAttr {{.*}} Implicit

__interface Declared;
// CHECK: CXXRecordDecl {{.*}} __interface Declared
// CHECK: MSNoVTableAttr {{.*}} Implicit

