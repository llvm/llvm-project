// RUN: %clang_cc1 -verify=gnu -std=gnu++11 %s
// RUN: %clang_cc1 -verify=expected,cxx11 -Wvla -std=gnu++11 %s
// RUN: %clang_cc1 -verify=expected,cxx11 -std=c++11 %s
// RUN: %clang_cc1 -verify=expected,cxx98 -std=c++98 %s
// RUN: %clang_cc1 -verify=expected,off -std=c++11 -Wno-vla-extension-static-assert %s
// gnu-no-diagnostics

// Demonstrate that we do not diagnose use of VLAs by default in GNU mode, but
// we do diagnose them in C++ mode. Also note that we suggest use of
// static_assert, but only in C++11 and later and only if the warning group is
// not disabled.

// C++98 mode does not emit the same notes as C++11 mode because in C++98,
// we're looking for an integer constant expression, whereas in C++11 and later,
// we're looking for a constant expression that is of integer type (these are
// different operations; ICE looks at the syntactic form of the expression, but
// C++11 constant expressions require calculating the expression value).
void func(int n) { // cxx11-note {{declared here}} off-note {{declared here}}
  int vla[n]; // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                 cxx11-note {{function parameter 'n' with unknown value cannot be used in a constant expression}} \
                 off-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
}

void old_style_static_assert(int n) { // cxx11-note 5 {{declared here}} off-note 2 {{declared here}}
  int array1[n != 12 ? 1 : -1]; // cxx11-warning {{variable length arrays in C++ are a Clang extension; did you mean to use 'static_assert'?}} \
                                   cxx98-warning {{variable length arrays in C++ are a Clang extension}} \
                                   cxx11-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
  int array2[n != 12 ? -1 : 1]; // cxx11-warning {{variable length arrays in C++ are a Clang extension; did you mean to use 'static_assert'?}} \
                                   cxx98-warning {{variable length arrays in C++ are a Clang extension}} \
                                   cxx11-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
  int array3[n != 12 ? 1 : n];  // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                                   cxx11-note {{function parameter 'n' with unknown value cannot be used in a constant expression}} \
                                   off-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
  int array4[(n ? 1 : -1)];     // cxx11-warning {{variable length arrays in C++ are a Clang extension; did you mean to use 'static_assert'?}} \
                                   cxx98-warning {{variable length arrays in C++ are a Clang extension}} \
                                   cxx11-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
  int array5[n ? 1 : 0];        // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                                   cxx11-note {{function parameter 'n' with unknown value cannot be used in a constant expression}} \
                                   off-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
}
