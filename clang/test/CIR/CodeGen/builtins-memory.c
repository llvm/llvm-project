// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir

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
