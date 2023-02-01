// RUN: fir-opt --array-value-copy %s | FileCheck --check-prefix=NOOPT %s
// RUN: fir-opt --array-value-copy="optimize-conflicts=true" %s | FileCheck --check-prefix=OPT %s

// Reproducer from SPEC CPU2017/527.cam4_r:
// module cam4
//   type, public :: pbuf_fld
//      real*8, pointer, dimension(:,:,:,:,:) :: fld_ptr
//   end type pbuf_fld
// contains
//   subroutine test(pbuf, ncol, lchnk, time_index, kvh_idx)
//     implicit none
//     interface
//        subroutine init(kvh)
//          real*8, intent(out) :: kvh(4,27)
//        end subroutine init
//     end interface
//     type(pbuf_fld), intent(inout), dimension(1000) :: pbuf
//     real*8 :: kvh(4,27)
//     integer kvh_idx, ncol, lchnk, time_index
//     call init(kvh)
//     pbuf(kvh_idx)%fld_ptr(1,1:ncol,1:27,lchnk,time_index) = kvh(:ncol,:)
//   end subroutine test
// end module cam4

// Verify that no memory is allocated for the temporary array,
// when compiling with optimizations:
// OPT-NOT: fir.allocmem
// NOOPT: fir.allocmem

module {
  func.func @_QMcam4Ptest(%arg0: !fir.ref<!fir.array<1000x!fir.type<_QMcam4Tpbuf_fld{fld_ptr:!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>}>>> {fir.bindc_name = "pbuf"}, %arg1: !fir.ref<i32> {fir.bindc_name = "ncol"}, %arg2: !fir.ref<i32> {fir.bindc_name = "lchnk"}, %arg3: !fir.ref<i32> {fir.bindc_name = "time_index"}, %arg4: !fir.ref<i32> {fir.bindc_name = "kvh_idx"}) {
    %c4 = arith.constant 4 : index
    %c27 = arith.constant 27 : index
    %0 = fir.alloca !fir.array<4x27xf64> {bindc_name = "kvh", uniq_name = "_QMcam4FtestEkvh"}
    fir.call @_QPinit(%0) fastmath<contract> : (!fir.ref<!fir.array<4x27xf64>>) -> ()
    %1 = fir.load %arg4 : !fir.ref<i32>
    %2 = fir.convert %1 : (i32) -> i64
    %c1_i64 = arith.constant 1 : i64
    %3 = arith.subi %2, %c1_i64 : i64
    %4 = fir.coordinate_of %arg0, %3 : (!fir.ref<!fir.array<1000x!fir.type<_QMcam4Tpbuf_fld{fld_ptr:!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>}>>>, i64) -> !fir.ref<!fir.type<_QMcam4Tpbuf_fld{fld_ptr:!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>}>>
    %5 = fir.field_index fld_ptr, !fir.type<_QMcam4Tpbuf_fld{fld_ptr:!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>}>
    %6 = fir.coordinate_of %4, %5 : (!fir.ref<!fir.type<_QMcam4Tpbuf_fld{fld_ptr:!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>>
    %7 = fir.load %6 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>>
    %c0 = arith.constant 0 : index
    %8:3 = fir.box_dims %7, %c0 : (!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>, index) -> (index, index, index)
    %c1 = arith.constant 1 : index
    %9:3 = fir.box_dims %7, %c1 : (!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>, index) -> (index, index, index)
    %c2 = arith.constant 2 : index
    %10:3 = fir.box_dims %7, %c2 : (!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>, index) -> (index, index, index)
    %c3 = arith.constant 3 : index
    %11:3 = fir.box_dims %7, %c3 : (!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>, index) -> (index, index, index)
    %12:3 = fir.box_dims %7, %c4 : (!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>, index) -> (index, index, index)
    %13 = fir.undefined index
    %14 = fir.convert %c1_i64 : (i64) -> index
    %15 = arith.subi %14, %8#0 : index
    %16 = fir.load %arg1 : !fir.ref<i32>
    %17 = fir.convert %16 : (i32) -> i64
    %18 = fir.convert %17 : (i64) -> index
    %19 = arith.subi %18, %14 : index
    %20 = arith.addi %19, %14 : index
    %21 = arith.divsi %20, %14 : index
    %22 = arith.cmpi sgt, %21, %c0 : index
    %23 = arith.select %22, %21, %c0 : index
    %c27_i64 = arith.constant 27 : i64
    %24 = fir.convert %c27_i64 : (i64) -> index
    %25 = arith.subi %24, %14 : index
    %26 = arith.addi %25, %14 : index
    %27 = arith.divsi %26, %14 : index
    %28 = arith.cmpi sgt, %27, %c0 : index
    %29 = arith.select %28, %27, %c0 : index
    %30 = fir.load %arg2 : !fir.ref<i32>
    %31 = fir.convert %30 : (i32) -> i64
    %32 = fir.convert %31 : (i64) -> index
    %33 = arith.subi %32, %11#0 : index
    %34 = fir.load %arg3 : !fir.ref<i32>
    %35 = fir.convert %34 : (i32) -> i64
    %36 = fir.convert %35 : (i64) -> index
    %37 = arith.subi %36, %12#0 : index
    %38 = fir.shift %8#0, %9#0, %10#0, %11#0, %12#0 : (index, index, index, index, index) -> !fir.shift<5>
    %39 = fir.slice %c1_i64, %13, %13, %14, %18, %14, %14, %24, %14, %31, %13, %13, %35, %13, %13 : (i64, index, index, index, index, index, index, index, index, i64, index, index, i64, index, index) -> !fir.slice<5>
    %40 = fir.array_load %7(%38) [%39] : (!fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>, !fir.shift<5>, !fir.slice<5>) -> !fir.array<?x?x?x?x?xf64>
    %41 = arith.addi %c1, %c27 : index
    %42 = arith.subi %41, %c1 : index
    %43 = fir.shape %c4, %c27 : (index, index) -> !fir.shape<2>
    %44 = fir.slice %c1, %18, %14, %c1, %42, %14 : (index, index, index, index, index, index) -> !fir.slice<2>
    %45 = fir.array_load %0(%43) [%44] : (!fir.ref<!fir.array<4x27xf64>>, !fir.shape<2>, !fir.slice<2>) -> !fir.array<4x27xf64>
    %46 = arith.subi %23, %c1 : index
    %47 = arith.subi %29, %c1 : index
    %48 = fir.do_loop %arg5 = %c0 to %47 step %c1 unordered iter_args(%arg6 = %40) -> (!fir.array<?x?x?x?x?xf64>) {
      %49 = fir.do_loop %arg7 = %c0 to %46 step %c1 unordered iter_args(%arg8 = %arg6) -> (!fir.array<?x?x?x?x?xf64>) {
        %50 = fir.array_fetch %45, %arg7, %arg5 : (!fir.array<4x27xf64>, index, index) -> f64
        %51 = fir.array_update %arg8, %50, %15, %arg7, %arg5, %33, %37 : (!fir.array<?x?x?x?x?xf64>, f64, index, index, index, index, index) -> !fir.array<?x?x?x?x?xf64>
        fir.result %51 : !fir.array<?x?x?x?x?xf64>
      }
      fir.result %49 : !fir.array<?x?x?x?x?xf64>
    }
    fir.array_merge_store %40, %48 to %7[%39] : !fir.array<?x?x?x?x?xf64>, !fir.array<?x?x?x?x?xf64>, !fir.box<!fir.ptr<!fir.array<?x?x?x?x?xf64>>>, !fir.slice<5>
    return
  }
  func.func private @_QPinit(!fir.ref<!fir.array<4x27xf64>>)
  fir.dispatch_table @_QMcam4Tpbuf_fld
}
