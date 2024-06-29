// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-linux-gnu -verify %s -DSUPPORTED=1
// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-apple-macosx -verify %s -DSUPPORTED=1
// RUN: %clang_cc1 -emit-llvm-only -triple arm64-apple-macosx -verify %s -DSUPPORTED=1
// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-pc-win32 -verify %s -DNOT_SUPPORTED=1

// expected-no-diagnostics

#if __has_attribute(ifunc)
#  if NOT_SUPPORTED
#    error "ifunc appears to be supported on this platform, but shouldn't be"
#  endif
#else
#  if SUPPORTED
#    error "ifunc should be supported on this platform, but isn't"
#  endif
#endif
