// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -F %S/Inputs -DFOO_RETURNS_INT_PTR -verify %s
// RUN: %clang_cc1 -fmodule-cache-path %t -F %S/Inputs -verify %s

__import_module__ CmdLine;

void test() {
#ifdef FOO_RETURNS_INT_PTR
  int *ip = foo();
#else
  float *fp = foo();
#endif
}
