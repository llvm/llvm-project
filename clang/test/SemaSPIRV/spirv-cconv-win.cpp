/// Tests that SPIRV64 accepts Windows calling conventions with Windows aux-triple.

// RUN: %clang_cc1 -fsycl-is-device \
// RUN:   -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

void __vectorcall vector_func(float x, float y) {}
void __regcall regcall_func(int x) {}
void __stdcall stdcall_func(int x) {}
void default_func(int x) {}

typedef void (__vectorcall *VecFnPtr)(float, float);
typedef void (__regcall *RegFnPtr)(int);

VecFnPtr vfp = &vector_func;
RegFnPtr rfp = &regcall_func;
