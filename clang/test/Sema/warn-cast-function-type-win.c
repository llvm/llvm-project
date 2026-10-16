// RUN: %clang_cc1 %s -triple x86_64-windows -fsyntax-only -Wcast-function-type -Wno-cast-function-type-strict -verify=windows
// RUN: %clang_cc1 %s -triple x86_64-windows -fsyntax-only -Wcast-function-type -Wno-cast-function-type-strict -x c++ -verify=windows
// RUN: %clang_cc1 %s -triple x86_64-pc-linux -fsyntax-only -Wcast-function-type -Wno-cast-function-type-strict -verify=linux
// RUN: %clang_cc1 %s -triple x86_64-pc-linux -fsyntax-only -Wcast-function-type -Wno-cast-function-type-strict -x c++ -verify=linux,linux-cpp
// RUN: %clang_cc1 %s -triple x86_64-windows -fsyntax-only -Wcast-function-type -Wcast-function-type-strict -x c++ -verify=strict
// windows-no-diagnostics

// On Windows targets, this is expected to compile fine, and on non-Windows
// targets, this should diagnose the mismatch. This is to allow for idiomatic
// use of GetProcAddress, similar to what we do for dlsym. On non-Windows
// targets, this should be diagnosed.
typedef int (*FARPROC1)();
typedef unsigned long long (*FARPROC2)();

FARPROC1 GetProcAddress1(void);
FARPROC2 GetProcAddress2(void);

typedef int (*test1_type)(int);
typedef float(*test2_type)();

void test(void) {
  // This does not diagnose on Linux in C mode because FARPROC1 has a matching
  // return type to test1_type, but FARPROC1 has no prototype and so checking
  // is disabled for further compatibility issues. In C++ mode, all functions
  // have a prototype and so the check happens.
  test1_type t1 = (test1_type)GetProcAddress1();
  // linux-cpp-warning@-1 {{cast from 'FARPROC1' (aka 'int (*)()') to 'test1_type' (aka 'int (*)(int)') converts to incompatible function type}}
  // strict-warning@-2 {{cast from 'FARPROC1' (aka 'int (*)()') to 'test1_type' (aka 'int (*)(int)') converts to incompatible function type}}
  
  // This case is diagnosed in both C and C++ modes on Linux because the return
  // type of FARPROC2 does not match the return type of test2_type.
  test2_type t2 = (test2_type)GetProcAddress2();
  // linux-warning@-1 {{cast from 'FARPROC2' (aka 'unsigned long long (*)()') to 'test2_type' (aka 'float (*)()') converts to incompatible function type}}
  // strict-warning@-2 {{cast from 'FARPROC2' (aka 'unsigned long long (*)()') to 'test2_type' (aka 'float (*)()') converts to incompatible function type}}
}

