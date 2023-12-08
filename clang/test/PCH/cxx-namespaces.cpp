// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx-namespaces.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/cxx-namespaces.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -ast-dump-lookups -ast-dump-filter N %s | FileCheck %s

// Test with modules.
// RUN: %clang_cc1 -fmodules -x c++-header -emit-pch -o %t %S/cxx-namespaces.h
// RUN: %clang_cc1 -fmodules -include-pch %t -fsyntax-only -verify %s
// RUN: %clang_cc1 -fmodules -include-pch %t -fsyntax-only -ast-dump-lookups -ast-dump-filter N %s | FileCheck %s

// expected-no-diagnostics

void m() {
  N::x = 0;
  N::f();
}

// namespace 'N' should contain only two declarations of 'f'.

// CHECK:      DeclarationName 'f'
// CHECK-NEXT: |-Function {{.*}} 'f' 'void (
// CHECK-NEXT: `-Function {{.*}} 'f' 'void (
