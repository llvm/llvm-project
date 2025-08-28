// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fenable-ripple -ast-print %s | FileCheck %s

#define ripple_set_block_shape(PEId, Size) \
  (ripple_block_t) __builtin_ripple_set_shape((PEId), (Size), 1, 1, 1, 1, 1, 1, 1, 1, 1)

typedef struct ripple_block_shape *ripple_block_t;

void test_one(int x) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
#pragma ripple parallel Block(BS) Dims(0)
// CHECK: #pragma ripple parallel Block(BS) Dims(0)
  for (int i = 0; i < x; i++)
    ;
}

void test_two(int x, int y) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
// CHECK: #pragma ripple parallel Block(BS) Dims(0)
#pragma ripple parallel Block(BS) Dims(0)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_three(int x, int y) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
// CHECK: #pragma ripple parallel Block(BS) Dims(0)
#pragma ripple parallel Block(BS) Dims(0)
  for (int i = 0; i < x; i++)
// CHECK: #pragma ripple parallel Block(BS) Dims(1)
#pragma ripple parallel Block(BS) Dims(1)
    for (int i = 0; i < y; i++)
      ;
}

void test_four(int x, int y) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  for (int i = 0; i < x; i++)
// CHECK: #pragma ripple parallel Block(BS) Dims(0) NoRemainder
#pragma ripple parallel NoRemainder Block(BS) Dims(0)
    for (int i = 0; i < y; i++)
      ;
}

void test_five(int x, int y, int z) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
// CHECK: #pragma ripple parallel Block(BS) Dims(0, 1)
#pragma ripple parallel Block(BS) Dims(0, 1)
  for (int i = 0; i < x; i++)
// CHECK: #pragma ripple parallel Block(BS) Dims(1, 0)
#pragma ripple parallel Block(BS) Dims(1, 0)
    for (int i = 0; i < y; i++)
// CHECK: #pragma ripple parallel Block(BS) Dims(0, 1) NoRemainder
#pragma ripple parallel Block(BS) Dims(0, 1) NoRemainder
      for (int i = 0; i < z; i++)
        ;
}
