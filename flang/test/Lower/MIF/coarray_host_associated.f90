! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program p
  integer :: c[*]
  c = 5
  if (this_image() == 1) call show
contains
  subroutine show
    c = c + 5
  end subroutine
end program

! CHECK-LABEL: func.func @_QQmain() attributes {fir.bindc_name = "P"} {
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
! CHECK: %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
! CHECK: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[VAL_3:.*]] = fir.address_of(@_QFEc) : !fir.ref<i32>
! CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! CHECK: fir.store %[[C1_I64]] to %[[VAL_4]] : !fir.ref<i64>
! CHECK: %[[VAL_5:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! CHECK: %[[VAL_6:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! CHECK: mif.alloc_coarray %[[VAL_3]] lcobounds %[[VAL_5]] ucobounds %[[VAL_6]] uniq_name("_QFEc") : (!fir.ref<i32>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
! CHECK: %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFEc") fortran_attrs<internal_assoc> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[C5_I32:.*]] = arith.constant 5 : i32
! CHECK: hlfir.assign %[[C5_I32]] to %[[VAL_7]]#0 : i32, !fir.ref<i32>
! CHECK: %[[VAL_8:.*]] = mif.this_image : () -> i32
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_8]], %[[C1_I32]] : i32
! CHECK: fir.if %[[VAL_9]] {
! CHECK:   fir.call @_QFPshow() fastmath<contract> : () -> ()
! CHECK: }
! CHECK: return
! CHECK: }

! mif.alloc_coarray must not appear in the body of @_QFPshow
! CHECK-LABEL: func.func private @_QFPshow() attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFEc) : !fir.ref<i32>
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFEc") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK: %[[C5_I32:.*]] = arith.constant 5 : i32
! CHECK: %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[C5_I32]] : i32
! CHECK: hlfir.assign %[[VAL_4]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
! CHECK: return
! CHECK: }

! CHECK: fir.global internal @_QFEc : i32 {
! CHECK:   %[[VAL_0:.*]] = fir.zero_bits i32
! CHECK:   fir.has_value %[[VAL_0]] : i32
! CHECK: }
