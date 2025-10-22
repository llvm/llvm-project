// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -O1 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

extern int global_var;

__attribute__((always_inline)) inline int always_inline_function(int x) {
  return x * 2 + global_var;
}

inline int inline_hint_function(int x) {
  return x - 1 + global_var;
}

__attribute__((noinline)) int noinline_function(int x) {
  return x / 2 + global_var;
}

int regular_function(int x) {
  return x + 1 + global_var;
}

// Force emission of all functions with function pointers
int (*always_inline_ptr)(int) = &always_inline_function;
int (*inline_hint_ptr)(int) = &inline_hint_function;
int (*noinline_ptr)(int) = &noinline_function;
int (*regular_ptr)(int) = &regular_function;

// CIR-LABEL: cir.func dso_local @_Z17noinline_functioni(%arg0: !s32i {{.*}}) -> !s32i inline(never)

// CIR-LABEL: cir.func dso_local @_Z16regular_functioni(%arg0: !s32i {{.*}}) -> !s32i
// CIR-NOT: inline(never)
// CIR-NOT: inline(always)
// CIR-NOT: inline(hint)
// CIR-SAME: {

// CIR-LABEL: cir.func {{.*}}@_Z22always_inline_functioni(%arg0: !s32i {{.*}}) -> !s32i inline(always)

// CIR-LABEL: cir.func {{.*}}@_Z20inline_hint_functioni(%arg0: !s32i {{.*}}) -> !s32i inline(hint)

// LLVM: ; Function Attrs:{{.*}} noinline
// LLVM: define{{.*}} i32 @_Z17noinline_functioni

// LLVM: ; Function Attrs:
// LLVM-NOT: noinline
// LLVM-NOT: alwaysinline
// LLVM-NOT: inlinehint
// LLVM-SAME: {{$}}
// LLVM: define{{.*}} i32 @_Z16regular_functioni

// LLVM: ; Function Attrs:{{.*}} alwaysinline
// LLVM: define{{.*}} i32 @_Z22always_inline_functioni

// LLVM: ; Function Attrs:{{.*}} inlinehint
// LLVM: define{{.*}} i32 @_Z20inline_hint_functioni

// OGCG: ; Function Attrs:{{.*}} noinline
// OGCG: define{{.*}} i32 @_Z17noinline_functioni

// OGCG: ; Function Attrs:
// OGCG-NOT: noinline
// OGCG-NOT: alwaysinline
// OGCG-NOT: inlinehint
// OGCG-SAME: {{$}}
// OGCG: define{{.*}} i32 @_Z16regular_functioni

// OGCG: ; Function Attrs:{{.*}} alwaysinline
// OGCG: define{{.*}} i32 @_Z22always_inline_functioni

// OGCG: ; Function Attrs:{{.*}} inlinehint
// OGCG: define{{.*}} i32 @_Z20inline_hint_functioni

