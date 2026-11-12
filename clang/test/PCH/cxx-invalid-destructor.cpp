// RUN: %clang_cc1 -x c++ -std=c++11 -emit-pch -o %t %S/cxx-invalid-destructor.h -fallow-pch-with-compiler-errors
// RUN: %clang_cc1 -x c++ -std=c++11 -include-pch %t %S/cxx-invalid-destructor.cpp -fsyntax-only -fno-validate-pch

Foo f;
