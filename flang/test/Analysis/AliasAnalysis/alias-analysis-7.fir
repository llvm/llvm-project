// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' 2>&1 | FileCheck %s

// leslie3d case with two allocatable module variables
// that cannot alias:
// module les3d_data
//   implicit real*8 (a-h,o-z)
//   integer imax, jmax, kmax
//   double precision,allocatable,dimension(:,:,:,:,:) :: q
//   double precision,allocatable,dimension(:,:,:,:) :: du
// end module les3d_data
// subroutine update()
//   use les3d_data
//   implicit real*8(a-h,o-z)
//   i2 = imax - 1
//   do k = 1, kmax - 1
//      do j = 1, jmax - 1
//         q(1:i2,j,k,1,m) = (q(1:i2,j,k,1,m) + du(1:i2,j,k,1))
//      end do
//   end do
// end subroutine update

// CHECK: allocatable_mod1#0 <-> allocatable_mod2#0: NoAlias
func.func @_QPupdate() {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.address_of(@_QMles3d_dataEdu) : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x?xf64>>>>
  %1:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMles3d_dataEdu"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x?xf64>>>>)
  %2 = fir.alloca i32 {bindc_name = "i2", uniq_name = "_QFupdateEi2"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFupdateEi2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.address_of(@_QMles3d_dataEimax) : !fir.ref<i32>
  %5:2 = hlfir.declare %4 {uniq_name = "_QMles3d_dataEimax"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFupdateEj"}
  %7:2 = hlfir.declare %6 {uniq_name = "_QFupdateEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %8 = fir.address_of(@_QMles3d_dataEjmax) : !fir.ref<i32>
  %9:2 = hlfir.declare %8 {uniq_name = "_QMles3d_dataEjmax"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %10 = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFupdateEk"}
  %11:2 = hlfir.declare %10 {uniq_name = "_QFupdateEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %12 = fir.address_of(@_QMles3d_dataEkmax) : !fir.ref<i32>
  %13:2 = hlfir.declare %12 {uniq_name = "_QMles3d_dataEkmax"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %14 = fir.alloca i32 {bindc_name = "m", uniq_name = "_QFupdateEm"}
  %15:2 = hlfir.declare %14 {uniq_name = "_QFupdateEm"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %16 = fir.address_of(@_QMles3d_dataEq) : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x?x?xf64>>>>
  %17:2 = hlfir.declare %16 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMles3d_dataEq"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x?x?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x?x?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x?x?xf64>>>>)
  %18 = fir.load %5#0 : !fir.ref<i32>
  %19 = arith.subi %18, %c1_i32 : i32
  hlfir.assign %19 to %3#0 : i32, !fir.ref<i32>
  %20 = fir.load %13#0 : !fir.ref<i32>
  %21 = arith.subi %20, %c1_i32 : i32
  %22 = fir.convert %21 : (i32) -> index
  %23 = fir.convert %c1 : (index) -> i32
  %24:2 = fir.do_loop %arg0 = %c1 to %22 step %c1 iter_args(%arg1 = %23) -> (index, i32) {
    fir.store %arg1 to %11#1 : !fir.ref<i32>
    %25 = fir.load %9#0 : !fir.ref<i32>
    %26 = arith.subi %25, %c1_i32 : i32
    %27 = fir.convert %26 : (i32) -> index
    %28:2 = fir.do_loop %arg2 = %c1 to %27 step %c1 iter_args(%arg3 = %23) -> (index, i32) {
      fir.store %arg3 to %7#1 : !fir.ref<i32>
      %32 = fir.load %17#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x?x?xf64>>>>
      %33 = fir.load %3#0 : !fir.ref<i32>
      %34 = fir.convert %33 : (i32) -> index
      %35 = arith.cmpi sgt, %34, %c0 : index
      %36 = arith.select %35, %34, %c0 : index
      %37 = fir.load %7#0 : !fir.ref<i32>
      %38 = fir.convert %37 : (i32) -> i64
      %39 = fir.load %11#0 : !fir.ref<i32>
      %40 = fir.convert %39 : (i32) -> i64
      %41 = fir.load %15#0 : !fir.ref<i32>
      %42 = fir.convert %41 : (i32) -> i64
      %43 = fir.shape %36 : (index) -> !fir.shape<1>
      %44 = hlfir.designate %32 (%c1:%34:%c1, %38, %40, %c1, %42)  shape %43 {test.ptr = "allocatable_mod1"} : (!fir.box<!fir.heap<!fir.array<?x?x?x?x?xf64>>>, index, index, index, i64, i64, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
      %45 = fir.load %1#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x?xf64>>>>
      %46 = hlfir.designate %45 (%c1:%34:%c1, %38, %40, %c1)  shape %43 {test.ptr = "allocatable_mod2"} : (!fir.box<!fir.heap<!fir.array<?x?x?x?xf64>>>, index, index, index, i64, i64, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
      %47 = hlfir.elemental %43 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf64> {
      ^bb0(%arg4: index):
        %51 = hlfir.designate %44 (%arg4)  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
        %52 = hlfir.designate %46 (%arg4)  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
        %53 = fir.load %51 : !fir.ref<f64>
        %54 = fir.load %52 : !fir.ref<f64>
        %55 = arith.addf %53, %54 fastmath<fast> : f64
        %56 = hlfir.no_reassoc %55 : f64
        hlfir.yield_element %56 : f64
      }
      hlfir.assign %47 to %44 : !hlfir.expr<?xf64>, !fir.box<!fir.array<?xf64>>
      hlfir.destroy %47 : !hlfir.expr<?xf64>
      %48 = arith.addi %arg2, %c1 : index
      %49 = fir.load %7#1 : !fir.ref<i32>
      %50 = arith.addi %49, %23 : i32
      fir.result %48, %50 : index, i32
    }
    fir.store %28#1 to %7#1 : !fir.ref<i32>
    %29 = arith.addi %arg0, %c1 : index
    %30 = fir.load %11#1 : !fir.ref<i32>
    %31 = arith.addi %30, %23 : i32
    fir.result %29, %31 : index, i32
  }
  fir.store %24#1 to %11#1 : !fir.ref<i32>
  return
}
