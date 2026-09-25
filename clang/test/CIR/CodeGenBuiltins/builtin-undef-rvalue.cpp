// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -verify %s -o - > %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

typedef float v4sf __attribute__((vector_size(16)));

float test_builtin_reduce_maximum_undef_rvalue(v4sf x) {
  // expected-error@+1 {{unimplemented builtin call: __builtin_reduce_maximum}}
  return __builtin_reduce_maximum(x);
}

// CIR-LABEL: test_builtin_reduce_maximum_undef_rvalue
// CIR:         cir.const #cir.undef : !cir.float
// CIR:         cir.return
