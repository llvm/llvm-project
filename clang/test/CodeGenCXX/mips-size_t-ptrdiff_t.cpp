// RUN: %clang_cc1 -x c++ -emit-llvm -triple=mips-unknown-linux-gnu < %s | FileCheck --check-prefix=O32 %s
// RUN: %clang_cc1 -x c++ -emit-llvm -triple=mips64-unknown-linux-gnu -target-abi n32 < %s | FileCheck --check-prefix=N32 %s
// RUN: %clang_cc1 -x c++ -emit-llvm -triple=mips64-unknown-linux-gnu -target-abi n64 < %s | FileCheck --check-prefix=N64 %s

// Test that the size_t is correct for the ABI. It's not sufficient to be the
// correct size, it must be the same type for correct name mangling.

long *alloc_long() {
  long *rv = new long; // size_t is implicit in the new operator
  return rv;
}
// O32-LABEL: define{{.*}} ptr @_Z10alloc_longv()
// O32: call noalias noundef nonnull ptr @_Znwj(i32 noundef signext 4)

// N32-LABEL: define{{.*}} ptr @_Z10alloc_longv()
// N32: call noalias noundef nonnull ptr @_Znwj(i32 noundef signext 4)

// N64-LABEL: define{{.*}} ptr @_Z10alloc_longv()
// N64: call noalias noundef nonnull ptr @_Znwm(i64 noundef zeroext 8)

long *alloc_long_array() {
  long *rv = new long[2];
  return rv;
}

// O32-LABEL: define{{.*}} ptr @_Z16alloc_long_arrayv()
// O32: call noalias noundef nonnull ptr @_Znaj(i32 noundef signext 8)

// N32-LABEL: define{{.*}} ptr @_Z16alloc_long_arrayv()
// N32: call noalias noundef nonnull ptr @_Znaj(i32 noundef signext 8)

// N64-LABEL: define{{.*}} ptr @_Z16alloc_long_arrayv()
// N64: call noalias noundef nonnull ptr @_Znam(i64 noundef zeroext 16)

#include <stddef.h>

void size_t_arg(size_t a) {
}

// O32-LABEL: _Z10size_t_argj
// N32-LABEL: _Z10size_t_argj
// N64-LABEL: _Z10size_t_argm

void ptrdiff_t_arg(ptrdiff_t a) {
}

// O32-LABEL: _Z13ptrdiff_t_argi
// N32-LABEL: _Z13ptrdiff_t_argi
// N64-LABEL: _Z13ptrdiff_t_argl
