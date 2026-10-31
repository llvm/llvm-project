// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -x c++ -emit-module-interface -o %t/A.pcm - < %t/a.cppm
// RUN: %clang_cc1 -std=c++20 -x c++ -emit-module-interface -o %t/B.pcm - < %t/b.cppm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=A=%t/A.pcm -fmodule-file=B=%t/B.pcm -fsyntax-only -verify %t/use.cpp
// RUN: %clang_cc1 -std=c++20 -fmodule-file=A=%t/A.pcm -fmodule-file=B=%t/B.pcm -x c++ -fsyntax-only -verify - < %t/use.cpp
//
// Test again with reduced BMI.
// RUN: %clang_cc1 -std=c++20 -x c++ -emit-reduced-module-interface -fmodules-embed-all-files -o %t/A.pcm - < %t/a.cppm
// RUN: %clang_cc1 -std=c++20 -x c++ -emit-reduced-module-interface -fmodules-embed-all-files -o %t/B.pcm - < %t/b.cppm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=A=%t/A.pcm -fmodule-file=B=%t/B.pcm -fsyntax-only -verify %t/use.cpp
// RUN: %clang_cc1 -std=c++20 -fmodule-file=A=%t/A.pcm -fmodule-file=B=%t/B.pcm -x c++ -fsyntax-only -verify - < %t/use.cpp

//--- a.cppm
export module A;
export template <class T> void fail_a(T t) { t.missing_a(); }

//--- b.cppm
export module B;

export template <class T> void fail_b(T t) { t.missing_b(); }

//--- use.cpp
import A;
import B;
void test() {
  fail_a(0); // expected-note {{in instantiation of function template specialization 'fail_a<int>' requested here}}
  fail_b(0); // expected-note {{in instantiation of function template specialization 'fail_b<int>' requested here}}
}

// expected-error@2 {{member reference base type 'int' is not a structure or union}}
// expected-error@3 {{member reference base type 'int' is not a structure or union}}
