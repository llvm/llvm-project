// RUN: %clang_cc1 -triple thumbv7-windows -fms-compatibility -fsyntax-only %s -verify
// expected-no-diagnostics

extern "C" {
typedef char * va_list;
void __va_start(va_list *, ...);
}

int test___va_start(int i, ...) {
  va_list ap;
  __va_start(&ap, ( &reinterpret_cast<const char &>(i) ),
             ( (sizeof(i) + 4 - 1) & ~(4 - 1) ),
             ( &reinterpret_cast<const char &>(i) ));
  return (*(int *)((ap += ( (sizeof(int) + 4 - 1) & ~(4 - 1) ) + ( ((va_list)0 - (ap)) & (__alignof(int) - 1) )) - ( (sizeof(int) + 4 - 1) & ~(4 - 1) )));
}

int builtin(int i, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, i);
  return __builtin_va_arg(ap, int);
}

