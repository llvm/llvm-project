// Test C++ profiles (P3589) enforced by C++20 named modules.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -fexperimental-cxx-profiles \
// RUN:   -emit-module-interface %t/A.cppm -o %t/A.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fexperimental-cxx-profiles \
// RUN:   -emit-module-interface %t/B.cppm -o %t/B.pcm -verify
//
// The implementation unit of A inherits the profiles A enforces on its module
// declaration.
// RUN: %clang_cc1 -std=c++20 -fexperimental-cxx-profiles \
// RUN:   -fmodule-file=A=%t/A.pcm -fsyntax-only %t/A-impl.cpp -verify
//
// Templates are checked according to the profiles of the module that defines
// them, not the importer.
// RUN: not %clang_cc1 -std=c++20 -fexperimental-cxx-profiles \
// RUN:   -fmodule-file=A=%t/A.pcm -fmodule-file=B=%t/B.pcm \
// RUN:   -fsyntax-only %t/use.cpp 2>&1 | FileCheck %t/use.cpp
// RUN: %clang_cc1 -std=c++20 -fexperimental-cxx-profiles \
// RUN:   -fmodule-file=A=%t/A.pcm -fmodule-file=B=%t/B.pcm \
// RUN:   -fsyntax-only %t/enforcing-user.cpp -verify

//--- A.cppm
// expected-no-diagnostics
export module A [[profiles::enforce(std::casting)]];

export char *a(char *p) { return p; }

export template <typename T> char *at(T *p) {
  return reinterpret_cast<char *>(p);
}

//--- B.cppm
// expected-no-diagnostics
export module B;

export template <typename T> char *bt(T *p) {
  return reinterpret_cast<char *>(p);
}

//--- A-impl.cpp
module A;

char *impl(int *p) {
  return reinterpret_cast<char *>(p);
  // expected-error@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'char *'}}
}

//--- use.cpp
import A [[profiles::require(std::casting)]];
import B [[profiles::require(std::casting)]];

char *use_a(int *p) { return at(p); }
char *use_b(int *p) { return bt(p); }

// CHECK: use.cpp:2:{{[0-9]+}}: error: module 'B' does not enforce profile 'std::casting'
// CHECK: A.cppm:7:{{[0-9]+}}: error: std::casting forbids reinterpret_cast from 'int *' to 'char *'
// CHECK-NOT: B.cppm{{.*}}error

//--- enforcing-user.cpp
[[profiles::enforce(std::casting)]];
import B;

// B does not enforce std::casting, so its template is not checked here.
char *use_b(int *p) { return bt(p); }

char *local(int *p) {
  return reinterpret_cast<char *>(p);
  // expected-error@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'char *'}}
}
