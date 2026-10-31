// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-cir -verify %s -o - > %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

typedef int v4si __attribute__((vector_size(16)));

v4si test_nyi_target_builtin_undef_rvalue(v4si a, v4si b, v4si c,
                                          unsigned char m) {
  // expected-error@+1 {{unimplemented X86 builtin call: __builtin_ia32_pternlogd128_maskz}}
  return __builtin_ia32_pternlogd128_maskz(a, b, c, 1, m);
}

// CIR-LABEL: test_nyi_target_builtin_undef_rvalue
// CIR:         cir.const #cir.undef : !cir.vector<4 x !s32i>
// CIR:         cir.return
