// RUN: %clang_cc1 -fms-extensions -triple i386-unknown-unknown  -x c++-header -emit-pch -o %t %S/cxx-ms-function-specialization-class-scope.h
// RUN: %clang_cc1 -fms-extensions -triple i386-unknown-unknown -include-pch %t -fsyntax-only -verify %s 
// expected-no-diagnostics


void test2()
{
   B<char> b(3);
   char* ptr;
   b.f(ptr);
   b.f<int>(99);
   b.f(100);
}

