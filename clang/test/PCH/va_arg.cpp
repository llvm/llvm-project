// Test this without pch.
// RUN: %clang_cc1 -triple=x86_64-unknown-freebsd7.0 -include %S/Inputs/va_arg.h %s -emit-llvm -o -
// REQUIRES: x86-registered-target

// Test with pch.
// RUN: %clang_cc1 -triple=x86_64-unknown-freebsd7.0 -emit-pch -x c++-header -o %t %S/Inputs/va_arg.h
// RUN: %clang_cc1 -triple=x86_64-unknown-freebsd7.0 -include-pch %t %s -emit-llvm -o -

typedef __SIZE_TYPE__ size_t;

extern "C" {
int vsnprintf(char * , size_t, const char * , va_list) ;
int __attribute__((ms_abi)) wvsprintfA(char *, const char *, __ms_va_list);
}

void f(char *buffer, unsigned count, const char* format, va_list argptr) {
  vsnprintf(buffer, count, format, argptr);
}

void g(char *buffer, const char *format, __ms_va_list argptr) {
  wvsprintfA(buffer, format, argptr);
}
