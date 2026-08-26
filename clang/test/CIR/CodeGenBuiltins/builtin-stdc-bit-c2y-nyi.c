// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c2y -fclangir -emit-cir -verify -DSTDC_ROTATE_LEFT %s -o -
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c2y -fclangir -emit-cir -verify -DSTDC_ROTATE_RIGHT %s -o -
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c2y -fclangir -emit-cir -verify -DSTDC_MEMREVERSE8 %s -o -
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c2y -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

unsigned stdc_rotate_left_ui(unsigned, unsigned);
unsigned stdc_rotate_right_ui(unsigned, unsigned);
unsigned stdc_memreverse8u32(unsigned);

#if !defined(STDC_ROTATE_RIGHT) && !defined(STDC_MEMREVERSE8)
unsigned test_stdc_rotate_left_ui(unsigned x) {
  return stdc_rotate_left_ui(x, 1); // expected-error {{ClangIR code gen Not Yet Implemented: unimplemented X86 builtin call: stdc_rotate_left_ui}}
}

// OGCG-LABEL: define{{.*}} i32 @test_stdc_rotate_left_ui(
// OGCG: call i32 @llvm.fshl.i32(
#endif

#if !defined(STDC_ROTATE_LEFT) && !defined(STDC_MEMREVERSE8)
unsigned test_stdc_rotate_right_ui(unsigned x) {
  return stdc_rotate_right_ui(x, 1); // expected-error {{ClangIR code gen Not Yet Implemented: unimplemented X86 builtin call: stdc_rotate_right_ui}}
}

// OGCG-LABEL: define{{.*}} i32 @test_stdc_rotate_right_ui(
// OGCG: call i32 @llvm.fshr.i32(
#endif

#if !defined(STDC_ROTATE_LEFT) && !defined(STDC_ROTATE_RIGHT)
unsigned test_stdc_memreverse8u32(unsigned x) {
  return stdc_memreverse8u32(x); // expected-error {{ClangIR code gen Not Yet Implemented: unimplemented X86 builtin call: stdc_memreverse8u32}}
}

// OGCG-LABEL: define{{.*}} i32 @test_stdc_memreverse8u32(
// OGCG: call i32 @llvm.bswap.i32(
#endif
