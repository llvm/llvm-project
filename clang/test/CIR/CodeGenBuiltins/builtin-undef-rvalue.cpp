// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -verify %s -o - > %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -verify %s -o - > %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef int v4si __attribute__((vector_size(16)));

int test_builtin_reduce_add_undef_rvalue(v4si x) {
  // expected-error@+1 {{unimplemented X86 builtin call: __builtin_reduce_add}}
  return __builtin_reduce_add(x);
}

// CIR-LABEL: @_Z36test_builtin_reduce_add_undef_rvalueDv4_i
// CIR:         cir.const #cir.undef : !s32i
// CIR:         cir.return

// LLVM-LABEL: @_Z36test_builtin_reduce_add_undef_rvalueDv4_i
// LLVM:         store i32 undef, ptr %{{.+}}, align 4
// LLVM:         ret i32 %{{.+}}
