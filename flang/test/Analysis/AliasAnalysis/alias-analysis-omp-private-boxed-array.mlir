// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// Reproduces a pattern where a PRIVATE array is boxed during OMP
// privatization, so accesses go through a fir.load of the box followed
// by hlfir.designate. The alias analysis must recognize that the loaded
// box data (from a private/Allocate source) does not alias with a
// dummy argument array.
//
// Fortran source:
//   SUBROUTINE mysub(grid, buf, n)
//     REAL(8), INTENT(IN) :: grid(10,10,4)
//     REAL(8) :: buf(4)
//     INTEGER, INTENT(IN) :: n
//     INTEGER :: i
//   !$OMP PARALLEL DO PRIVATE(buf)
//     DO i = 1, n
//       buf(:) = grid(1, 1, :)
//     ENDDO
//   END SUBROUTINE

// CHECK-LABEL: Testing : "test_boxed_private_vs_arg"
// CHECK: arg_designate#0 <-> private_designate#0: NoAlias

omp.private {type = private} @buf_privatizer : !fir.box<!fir.array<4xf64>>

func.func @test_boxed_private_vs_arg(
    %arg0: !fir.ref<!fir.array<10x10x4xf64>> {fir.bindc_name = "grid"},
    %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10, %c10, %c4 : (index, index, index) -> !fir.shape<3>
  %1:2 = hlfir.declare %arg0(%0) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFmysubEgrid"} : (!fir.ref<!fir.array<10x10x4xf64>>, !fir.shape<3>) -> (!fir.ref<!fir.array<10x10x4xf64>>, !fir.ref<!fir.array<10x10x4xf64>>)
  %2 = fir.alloca !fir.array<4xf64> {bindc_name = "buf", uniq_name = "_QFmysubEbuf"}
  %3 = fir.shape %c4 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %2(%3) {uniq_name = "_QFmysubEbuf"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)

  %5 = fir.alloca !fir.box<!fir.array<4xf64>>
  %6 = fir.embox %4#0(%3) : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<4xf64>>
  fir.store %6 to %5 : !fir.ref<!fir.box<!fir.array<4xf64>>>

  omp.parallel private(@buf_privatizer %5 -> %arg2 : !fir.ref<!fir.box<!fir.array<4xf64>>>) {
    %10:2 = hlfir.declare %arg2 {uniq_name = "_QFmysubEbuf"} : (!fir.ref<!fir.box<!fir.array<4xf64>>>) -> (!fir.ref<!fir.box<!fir.array<4xf64>>>, !fir.ref<!fir.box<!fir.array<4xf64>>>)

    // Designate into the dummy argument array: grid(1, 1, 1:4)
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c4_1 = arith.constant 4 : index
    %c1_2 = arith.constant 1 : index
    %11 = fir.shape %c4_1 : (index) -> !fir.shape<1>
    %c1_i64 = arith.constant 1 : i64
    %12 = hlfir.designate %1#0 (%c1_i64, %c1_i64, %c1:%c4_1:%c1_2) shape %11 {test.ptr = "arg_designate"} : (!fir.ref<!fir.array<10x10x4xf64>>, i64, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<4xf64>>

    // Designate through loaded private box: buf(1:4)
    %13 = fir.load %10#0 : !fir.ref<!fir.box<!fir.array<4xf64>>>
    %c0 = arith.constant 0 : index
    %14:3 = fir.box_dims %13, %c0 : (!fir.box<!fir.array<4xf64>>, index) -> (index, index, index)
    %15 = arith.addi %14#0, %14#1 : index
    %16 = arith.subi %15, %c1 : index
    %c4_3 = arith.constant 4 : index
    %17 = fir.shape %c4_3 : (index) -> !fir.shape<1>
    %18 = hlfir.designate %13 (%14#0:%16:%c1_0) shape %17 {test.ptr = "private_designate"} : (!fir.box<!fir.array<4xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>

    omp.terminator
  }
  return
}
