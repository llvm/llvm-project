// Test the setup without incremental extensions first
// RUN: %clang_cc1 -std=c++17 -fdelayed-template-parsing -fpch-instantiate-templates %s -emit-pch -o %t.pch -verify
// RUN: %clang_cc1 -std=c++17 -fdelayed-template-parsing -include-pch %t.pch %s -verify

// RUN: %clang_cc1 -std=c++17 -fdelayed-template-parsing -fincremental-extensions -fpch-instantiate-templates %s -emit-pch -o %t.incremental.pch -verify
// RUN: %clang_cc1 -std=c++17 -fdelayed-template-parsing -fincremental-extensions -include-pch %t.incremental.pch %s -verify

// expected-no-diagnostics

#ifndef PCH
#define PCH

// Have one template that is instantiated in the PCH (via the passed option
// -fpch-instantiate-templates) and then serialized
template <typename T> T ft1() { return 0; }
inline int f1() { return ft1<int>(); }

// Have a second late-parsed template that needs to be deserialized
template <typename T> T ft2() { return 0; }

#else

int f2() { return ft2<int>(); }

#endif
