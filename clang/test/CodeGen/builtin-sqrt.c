// RUN: %clang_cc1 -fmath-errno -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s --check-prefix=HAS_ERRNO
// RUN: %clang_cc1              -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s --check-prefix=NO_ERRNO

// FIXME: If a builtin is supposed to have identical semantics to its libm twin, then it
// should not be marked "constant" in Builtins.def because that means it can't set errno.
// Note that both runs have 'readnone' on the libcall here.

float foo(float X) {
  // HAS_ERRNO: call float @sqrtf(float
  // NO_ERRNO: call float @sqrtf(float
  return __builtin_sqrtf(X);
}

// HAS_ERRNO: declare float @sqrtf(float) [[ATTR:#[0-9]+]]
// HAS_ERRNO: attributes [[ATTR]] = { nounwind readnone {{.*}}}

// NO_ERRNO: declare float @sqrtf(float) [[ATTR:#[0-9]+]]
// NO_ERRNO: attributes [[ATTR]] = { nounwind readnone {{.*}}}

