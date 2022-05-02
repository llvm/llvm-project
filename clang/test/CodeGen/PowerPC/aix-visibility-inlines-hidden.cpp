// RUN: %clang_cc1 -triple powerpc-unknown-aix -mcmodel=large -emit-llvm -o - -x c++ %s  | \
// RUN: FileCheck -check-prefix=VISIBILITY-IR %s

// RUN: %clang_cc1 -triple powerpc-unknown-aix -mcmodel=large \
// RUN:            -fvisibility-inlines-hidden -emit-llvm -o - -x c++ %s  | \
// RUN: FileCheck -check-prefixes=VISIBILITY-IR,HIDDENINLINE-IR %s

// RUN: %clang_cc1 -triple powerpc-unknown-aix -mcmodel=large -fvisibility-inlines-hidden \
// RUN:            -fvisibility default -emit-llvm -o - -x c++ %s  | \
// RUN: FileCheck -check-prefixes=VISIBILITY-IR,HIDDENINLINE-IR %s

// RUN: %clang_cc1 -triple powerpc-unknown-aix -mcmodel=large -mignore-xcoff-visibility -emit-llvm \
// RUN:            -fvisibility-inlines-hidden -fvisibility default -o - -x c++ %s  | \
// RUN: FileCheck -check-prefix=NOVISIBILITY-IR %s

int x = 66;
__attribute__((__noinline__)) inline void f() {
  x = 55;
}

#pragma GCC visibility push(hidden)
__attribute__((__noinline__)) inline void foo() {
  x = 55;
}
#pragma GCC visibility pop

int bar() {
  f();
  foo();
  return x;
}

// HIDDENINLINE-IR:     define linkonce_odr hidden void @_Z1fv()
// NOVISIBILITY-IR:   define linkonce_odr void @_Z1fv()

// VISIBILITY-IR:     define linkonce_odr hidden void @_Z3foov()
// NOVISIBILITY-IR:   define linkonce_odr void @_Z3foov()
