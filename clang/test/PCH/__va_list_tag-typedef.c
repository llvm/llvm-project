// This test checks the patch for the compilation error / crash described in D18557.

// Test as a C source
// RUN: %clang_cc1 -emit-pch -x c-header -o %t %S/Inputs/__va_list_tag-typedef.h
// RUN: %clang_cc1 -fsyntax-only -include-pch %t %s

// Test as a C++ source
// RUN: %clang_cc1 -emit-pch -x c++-header -o %t %S/Inputs/__va_list_tag-typedef.h
// RUN: %clang_cc1 -x c++ -fsyntax-only -include-pch %t %s

// expected-no-diagnostics

typedef __builtin_va_list va_list_2;
void test(const char* format, ...) { va_list args; va_start( args, format ); }
