// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// A PRIVATE array is unboxed (a plain fir.array) and its hlfir.declare is
// nested inside the omp.loop_nest of an omp.wsloop. The alias analysis must
// resolve the clause-carrying omp.wsloop from the private block argument's
// owner region -- not the declare's immediate parent (the omp.loop_nest) --
// so that the private is recognized as an Allocate source that does not alias
// a dummy argument array.
//
// Fortran source:
//   subroutine test(a, n)
//     real(8) :: a(10)
//     integer :: n, i
//     real(8) :: xx(3)
//   !$omp parallel do private(xx)
//     do i = 1, n
//       xx(1:3) = a(1:3)
//     enddo
//   end subroutine

// CHECK-LABEL: Testing : "test_unboxed_private_wsloop_vs_arg"
// CHECK: arg_designate#0 <-> private_designate#0: NoAlias

omp.private {type = private} @xx_privatizer : !fir.array<3xf64>

func.func @test_unboxed_private_wsloop_vs_arg(
    %arg0: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"},
    %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  %sh_a = fir.shape %c10 : (index) -> !fir.shape<1>
  %adecl:2 = hlfir.declare %arg0(%sh_a) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEa"} : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf64>>, !fir.ref<!fir.array<10xf64>>)
  %xx = fir.alloca !fir.array<3xf64> {bindc_name = "xx", uniq_name = "_QFtestExx"}
  %sh3 = fir.shape %c3 : (index) -> !fir.shape<1>
  %n = fir.load %arg1 : !fir.ref<i32>

  omp.wsloop private(@xx_privatizer %xx -> %parg : !fir.ref<!fir.array<3xf64>>) {
    omp.loop_nest (%iv) : i32 = (%c1_i32) to (%n) inclusive step (%c1_i32) {
      %xdecl:2 = hlfir.declare %parg(%sh3) {uniq_name = "_QFtestExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
      %ad = hlfir.designate %adecl#0 (%c1:%c3:%c1) shape %sh3 {test.ptr = "arg_designate"} : (!fir.ref<!fir.array<10xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
      %pd = hlfir.designate %xdecl#0 (%c1:%c3:%c1) shape %sh3 {test.ptr = "private_designate"} : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
      omp.yield
    }
  }
  return
}
