// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -verify %s -o - > %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

typedef int v4si __attribute__((vector_size(16)));

int test_builtin_reduce_add_undef_rvalue(v4si x) {
  // expected-error@+1 {{unimplemented builtin call: __builtin_reduce_add}}
  return __builtin_reduce_add(x);
}

// CIR-LABEL: @_Z36test_builtin_reduce_add_undef_rvalueDv4_i
// CIR:         cir.const #cir.undef : !s32i
// CIR:         cir.return
