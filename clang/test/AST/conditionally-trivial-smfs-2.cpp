// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-linux %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-win32 %s -ast-dump | FileCheck %s

template<class X>
struct DefaultConstructibleWithTemplate {
    template<class T = int>
    DefaultConstructibleWithTemplate();
};

void f() {
    DefaultConstructibleWithTemplate<int> x;
}

// CHECK: | `-ClassTemplateSpecializationDecl {{.*}} struct DefaultConstructibleWithTemplate definition
// CHECK: |   | |-CXXConstructorDecl {{.*}} DefaultConstructibleWithTemplate 'void ()'
