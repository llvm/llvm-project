// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - \
// RUN:  | opt -S -passes=instcombine,mem2reg,simplifycfg -o %t.ll 
// RUN: FileCheck  --check-prefix=LLVM --input-file=%t.ll %s

typedef __SIZE_TYPE__ size_t;
void test_memcpy_chk(void *dest, const void *src, size_t n) {
  // CIR-LABEL: cir.func @test_memcpy_chk
  // CIR:         %[[#DEST:]] = cir.alloca {{.*}} ["dest", init]
  // CIR:         %[[#SRC:]] = cir.alloca {{.*}} ["src", init]
  // CIR:         %[[#N:]] = cir.alloca {{.*}} ["n", init]

  // An unchecked memcpy should be emitted when the count and buffer size are
  // constants and the count is less than or equal to the buffer size.

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#SRC_LOAD:]] = cir.load %[[#SRC]]
  // CIR: %[[#COUNT:]] = cir.const #cir.int<8>
  // CIR: cir.libc.memcpy %[[#COUNT]] bytes from %[[#SRC_LOAD]] to %[[#DEST_LOAD]]
  __builtin___memcpy_chk(dest, src, 8, 10);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#SRC_LOAD:]] = cir.load %[[#SRC]]
  // CIR: %[[#COUNT:]] = cir.const #cir.int<10>
  // CIR: cir.libc.memcpy %[[#COUNT]] bytes from %[[#SRC_LOAD]] to %[[#DEST_LOAD]]
  __builtin___memcpy_chk(dest, src, 10, 10);

  // __memcpy_chk should be called when the count is greater than the buffer
  // size, or when either the count or buffer size isn't a constant.

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#SRC_LOAD:]] = cir.load %[[#SRC]]
  // CIR: %[[#COUNT:]] = cir.const #cir.int<10>
  // CIR: %[[#SIZE:]] = cir.const #cir.int<8>
  // CIR: cir.call @__memcpy_chk(%[[#DEST_LOAD]], %[[#SRC_LOAD]], %[[#COUNT]], %[[#SIZE]])
  __builtin___memcpy_chk(dest, src, 10lu, 8lu);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#SRC_LOAD:]] = cir.load %[[#SRC]]
  // CIR: %[[#N_LOAD:]] = cir.load %[[#N]]
  // CIR: %[[#SIZE:]] = cir.const #cir.int<10>
  // CIR: cir.call @__memcpy_chk(%[[#DEST_LOAD]], %[[#SRC_LOAD]], %[[#N_LOAD]], %[[#SIZE]])
  __builtin___memcpy_chk(dest, src, n, 10lu);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#SRC_LOAD:]] = cir.load %[[#SRC]]
  // CIR: %[[#COUNT:]] = cir.const #cir.int<10>
  // CIR: %[[#N_LOAD:]] = cir.load %[[#N]]
  // CIR: cir.call @__memcpy_chk(%[[#DEST_LOAD]], %[[#SRC_LOAD]], %[[#COUNT]], %[[#N_LOAD]])
  __builtin___memcpy_chk(dest, src, 10lu, n);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#SRC_LOAD:]] = cir.load %[[#SRC]]
  // CIR: %[[#N_LOAD1:]] = cir.load %[[#N]]
  // CIR: %[[#N_LOAD2:]] = cir.load %[[#N]]
  // CIR: cir.call @__memcpy_chk(%[[#DEST_LOAD]], %[[#SRC_LOAD]], %[[#N_LOAD1]], %[[#N_LOAD2]])
  __builtin___memcpy_chk(dest, src, n, n);
}

void test_memmove_chk(void *dest, const void *src, size_t n) {
  // CIR-LABEL: cir.func @test_memmove_chk
  // CIR:         %[[#DEST:]] = cir.alloca {{.*}} ["dest", init]
  // CIR:         %[[#SRC:]] = cir.alloca {{.*}} ["src", init]
  // CIR:         %[[#N:]] = cir.alloca {{.*}} ["n", init]

  // LLVM-LABEL: test_memmove_chk

  // An unchecked memmove should be emitted when the count and buffer size are
  // constants and the count is less than or equal to the buffer size.

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#SRC_LOAD:]] = cir.load %[[#SRC]]
  // CIR: %[[#COUNT:]] = cir.const #cir.int<8>
  // CIR: cir.libc.memmove %[[#COUNT]] bytes from %[[#SRC_LOAD]] to %[[#DEST_LOAD]]
  // LLVM: call void @llvm.memmove.p0.p0.i64(ptr {{%.*}}, ptr {{%.*}}, i64 8, i1 false)
  // COM: LLVM: call void @llvm.memmove.p0.p0.i64(ptr align 1 {{%.*}}, ptr align 1 {{%.*}}, i64 8, i1 false)
  __builtin___memmove_chk(dest, src, 8, 10);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#SRC_LOAD:]] = cir.load %[[#SRC]]
  // CIR: %[[#COUNT:]] = cir.const #cir.int<10>
  // CIR: cir.libc.memmove %[[#COUNT]] bytes from %[[#SRC_LOAD]] to %[[#DEST_LOAD]]
  // LLVM: call void @llvm.memmove.p0.p0.i64(ptr {{%.*}}, ptr {{%.*}}, i64 10, i1 false)
  // COM: LLVM: call void @llvm.memmove.p0.p0.i64(ptr align 1 {{%.*}}, ptr align 1 {{%.*}}, i64 10, i1 false)
  __builtin___memmove_chk(dest, src, 10, 10);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#SRC_LOAD:]] = cir.load %[[#SRC]]
  // CIR: %[[#COUNT:]] = cir.const #cir.int<10>
  // CIR: %[[#SIZE:]] = cir.const #cir.int<8>
  // CIR: cir.call @__memmove_chk(%[[#DEST_LOAD]], %[[#SRC_LOAD]], %[[#COUNT]], %[[#SIZE]])
  // LLVM: call ptr @__memmove_chk(ptr {{%.*}}, ptr {{%.*}}, i64 10, i64 8)
  // COM: LLVM: call ptr @__memmove_chk(ptr noundef %4, ptr noundef %5, i64 noundef 10, i64 noundef 8)
  __builtin___memmove_chk(dest, src, 10lu, 8lu);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#SRC_LOAD:]] = cir.load %[[#SRC]]
  // CIR: %[[#N_LOAD:]] = cir.load %[[#N]]
  // CIR: %[[#SIZE:]] = cir.const #cir.int<10>
  // CIR: cir.call @__memmove_chk(%[[#DEST_LOAD]], %[[#SRC_LOAD]], %[[#N_LOAD]], %[[#SIZE]])
  // LLVM: call ptr @__memmove_chk(ptr {{%.*}}, ptr {{%.*}}, i64 {{%.*}}, i64 10)
  // COM: LLVM: call ptr @__memmove_chk(ptr noundef {{%.*}}, ptr noundef {{%.*}}, i64 noundef {{%.*}}, i64 noundef 10)
  __builtin___memmove_chk(dest, src, n, 10lu);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#SRC_LOAD:]] = cir.load %[[#SRC]]
  // CIR: %[[#COUNT:]] = cir.const #cir.int<10>
  // CIR: %[[#N_LOAD:]] = cir.load %[[#N]]
  // CIR: cir.call @__memmove_chk(%[[#DEST_LOAD]], %[[#SRC_LOAD]], %[[#COUNT]], %[[#N_LOAD]])
  // LLVM: call ptr @__memmove_chk(ptr {{%.*}}, ptr {{%.*}}, i64 10, i64 {{%.*}})
  // COM: LLVM: call ptr @__memmove_chk(ptr noundef {{%.*}}, ptr noundef {{%.*}}, i64 noundef 10, i64 noundef {{%.*}})
  __builtin___memmove_chk(dest, src, 10lu, n);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#SRC_LOAD:]] = cir.load %[[#SRC]]
  // CIR: %[[#N_LOAD1:]] = cir.load %[[#N]]
  // CIR: %[[#N_LOAD2:]] = cir.load %[[#N]]
  // CIR: cir.call @__memmove_chk(%[[#DEST_LOAD]], %[[#SRC_LOAD]], %[[#N_LOAD1]], %[[#N_LOAD2]])
  // LLVM: call ptr @__memmove_chk(ptr {{%.*}}, ptr {{%.*}}, i64 {{%.*}}, i64 {{%.*}})
  // COM: LLVM: call ptr @__memmove_chk(ptr noundef {{%.*}}, ptr noundef {{%.*}}, i64 noundef {{%.*}}, i64 noundef {{%.*}})
  __builtin___memmove_chk(dest, src, n, n);
}


void test_memset_chk(void *dest, int ch, size_t n) {
  // CIR-LABEL: cir.func @test_memset_chk
  // CIR:         %[[#DEST:]] = cir.alloca {{.*}} ["dest", init]
  // CIR:         %[[#CH:]] = cir.alloca {{.*}} ["ch", init]
  // CIR:         %[[#N:]] = cir.alloca {{.*}} ["n", init]

  // An unchecked memset should be emitted when the count and buffer size are
  // constants and the count is less than or equal to the buffer size.

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#CH_LOAD:]] = cir.load %[[#CH]]
  // CIR: %[[#COUNT:]] = cir.const #cir.int<8>
  // CIR: cir.libc.memset %[[#COUNT]] bytes from %[[#DEST_LOAD]] set to %[[#CH_LOAD]]
  __builtin___memset_chk(dest, ch, 8, 10);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#CH_LOAD:]] = cir.load %[[#CH]]
  // CIR: %[[#COUNT:]] = cir.const #cir.int<10>
  // CIR: cir.libc.memset %[[#COUNT]] bytes from %[[#DEST_LOAD]] set to %[[#CH_LOAD]]
  __builtin___memset_chk(dest, ch, 10, 10);

  // __memset_chk should be called when the count is greater than the buffer
  // size, or when either the count or buffer size isn't a constant.

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#CH_LOAD:]] = cir.load %[[#CH]]
  // CIR: %[[#COUNT:]] = cir.const #cir.int<10>
  // CIR: %[[#SIZE:]] = cir.const #cir.int<8>
  // CIR: cir.call @__memset_chk(%[[#DEST_LOAD]], %[[#CH_LOAD]], %[[#COUNT]], %[[#SIZE]])
  __builtin___memset_chk(dest, ch, 10lu, 8lu);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#CH_LOAD:]] = cir.load %[[#CH]]
  // CIR: %[[#N_LOAD:]] = cir.load %[[#N]]
  // CIR: %[[#SIZE:]] = cir.const #cir.int<10>
  // CIR: cir.call @__memset_chk(%[[#DEST_LOAD]], %[[#CH_LOAD]], %[[#N_LOAD]], %[[#SIZE]])
  __builtin___memset_chk(dest, ch, n, 10lu);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#CH_LOAD:]] = cir.load %[[#CH]]
  // CIR: %[[#COUNT:]] = cir.const #cir.int<10>
  // CIR: %[[#N_LOAD:]] = cir.load %[[#N]]
  // CIR: cir.call @__memset_chk(%[[#DEST_LOAD]], %[[#CH_LOAD]], %[[#COUNT]], %[[#N_LOAD]])
  __builtin___memset_chk(dest, ch, 10lu, n);

  // CIR: %[[#DEST_LOAD:]] = cir.load %[[#DEST]]
  // CIR: %[[#CH_LOAD:]] = cir.load %[[#CH]]
  // CIR: %[[#N_LOAD1:]] = cir.load %[[#N]]
  // CIR: %[[#N_LOAD2:]] = cir.load %[[#N]]
  // CIR: cir.call @__memset_chk(%[[#DEST_LOAD]], %[[#CH_LOAD]], %[[#N_LOAD1]], %[[#N_LOAD2]])
  __builtin___memset_chk(dest, ch, n, n);
}

// FIXME: The test should test intrinsic argument alignment, however, 
// currently we lack support for argument attributes. 
// Thus, added `COM: LLVM:` lines so we can easily flip the test 
// when the support of argument attributes is in.
void test_memcpy_inline(void *dst, const void *src, size_t n) {

  // CIR-LABEL: test_memcpy_inline
  // CIR: cir.memcpy_inline 0 bytes from {{%.*}} to {{%.*}} : !cir.ptr<!void> -> !cir.ptr<!void>

  // LLVM-LABEL: test_memcpy_inline
  // LLVM: call void @llvm.memcpy.inline.p0.p0.i64(ptr {{%.*}}, ptr {{%.*}}, i64 0, i1 false)
  // COM: LLVM: call void @llvm.memcpy.inline.p0.p0.i64(ptr align 1 {{%.*}}, ptr align 1 {{%.*}}, i64 0, i1 false)
  __builtin_memcpy_inline(dst, src, 0);

  // CIR: cir.memcpy_inline 1 bytes from {{%.*}} to {{%.*}} : !cir.ptr<!void> -> !cir.ptr<!void>

  // LLVM: call void @llvm.memcpy.inline.p0.p0.i64(ptr {{%.*}}, ptr {{%.*}}, i64 1, i1 false)
  // COM: LLVM: call void @llvm.memcpy.inline.p0.p0.i64(ptr align 1 {{%.*}}, ptr align 1 {{%.*}}, i64 1, i1 false)
  __builtin_memcpy_inline(dst, src, 1);

  // CIR: cir.memcpy_inline 4 bytes from {{%.*}} to {{%.*}} : !cir.ptr<!void> -> !cir.ptr<!void>

  // LLVM: call void @llvm.memcpy.inline.p0.p0.i64(ptr {{%.*}}, ptr {{%.*}}, i64 4, i1 false)
  // COM: LLVM: call void @llvm.memcpy.inline.p0.p0.i64(ptr align 1 {{%.*}}, ptr align 1 {{%.*}}, i64 4, i1 false)
  __builtin_memcpy_inline(dst, src, 4);
}
 
void test_memcpy_inline_aligned_buffers(unsigned long long *dst, const unsigned long long *src) {

  // LLVM-LABEL: test_memcpy_inline_aligned_buffers
  // LLVM: call void @llvm.memcpy.inline.p0.p0.i64(ptr {{%.*}}, ptr {{%.*}}, i64 4, i1 false)
  // COM: LLVM: call void @llvm.memcpy.inline.p0.p0.i64(ptr align 8 {{%.*}}, ptr align 8 {{%.*}}, i64 4, i1 false)
  __builtin_memcpy_inline(dst, src, 4);
}

void test_memset_inline(void *dst, int val) {

  // CIR-LABEL: test_memset_inline
  // CIR: cir.memset_inline 0 bytes from {{%.*}} set to {{%.*}} : !cir.ptr<!void>, !s32i

  // LLVM-LABEL: test_memset_inline
  // LLVM: call void @llvm.memset.inline.p0.i64(ptr {{%.*}}, i8 {{%.*}}, i64 0, i1 false)
  __builtin_memset_inline(dst, val, 0);

  // CIR: cir.memset_inline 1 bytes from {{%.*}} set to {{%.*}} : !cir.ptr<!void>, !s32i

  // LLVM: call void @llvm.memset.inline.p0.i64(ptr {{%.*}}, i8 {{%.*}}, i64 1, i1 false)
  __builtin_memset_inline(dst, val, 1);

  // CIR: cir.memset_inline 4 bytes from {{%.*}} set to {{%.*}} : !cir.ptr<!void>, !s32i

  // LLVM: call void @llvm.memset.inline.p0.i64(ptr {{%.*}}, i8 {{%.*}}, i64 4, i1 false)
  __builtin_memset_inline(dst, val, 4);
}
