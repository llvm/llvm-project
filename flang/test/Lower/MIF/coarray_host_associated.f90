! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

module m
  integer, private :: cm[*]
contains
  integer function ub(d)
    integer, intent(in) :: d
    ub = ucobound(cm, dim=d)
  end function
end module

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
! CHECK-NEXT: %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
! CHECK-NEXT: %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
! CHECK-NEXT: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK-NEXT: %[[VAL_3:.*]] = fir.address_of(@_QFEc) : !fir.ref<i32>
! CHECK-NEXT: %[[C1_I64:.*]] = arith.constant 1 : i64
! CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
! CHECK-NEXT: %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! CHECK-NEXT: fir.store %[[C1_I64]] to %[[VAL_4]] : !fir.ref<i64>
! CHECK-NEXT: %[[VAL_5:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! CHECK-NEXT: %[[VAL_6:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! CHECK-NEXT: mif.alloc_coarray %[[VAL_3]] lcobounds %[[VAL_5]] ucobounds %[[VAL_6]] uniq_name("_QFEc") : (!fir.ref<i32>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
! CHECK-NEXT: %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFEc") fortran_attrs<internal_assoc> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-NEXT: %[[C5_I32:.*]] = arith.constant 5 : i32
! CHECK-NEXT: hlfir.assign %[[C5_I32]] to %[[VAL_7]]#0 : i32, !fir.ref<i32>
! CHECK-NEXT: %[[VAL_8:.*]] = mif.this_image : () -> i32
! CHECK-NEXT: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK-NEXT: %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_8]], %[[C1_I32]] : i32
! CHECK-NEXT: fir.if %[[VAL_9]] {
! CHECK-NEXT:   fir.call @_QFPshow() fastmath<contract> : () -> ()
! CHECK-NEXT: }
! CHECK-NEXT: return
! CHECK-NEXT: }

! mif.alloc_coarray must not appear in the body of @_QFPshow
! CHECK-LABEL: func.func private @_QFPshow() attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
! CHECK-NOT: mif.alloc_coarray
! CHECK-NEXT: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK-NEXT: %[[VAL_1:.*]] = fir.address_of(@_QFEc) : !fir.ref<i32>
! CHECK-NEXT: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFEc") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-NEXT: %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK-NEXT: %[[C5_I32:.*]] = arith.constant 5 : i32
! CHECK-NEXT: %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[C5_I32]] : i32
! CHECK-NEXT: hlfir.assign %[[VAL_4]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
! CHECK-NEXT: return
! CHECK-NEXT: }

! CHECK: fir.global @_QMmEcm : i32 {
! CHECK-NEXT:   %[[VAL_0:.*]] = fir.zero_bits i32
! CHECK-NEXT:   fir.has_value %[[VAL_0]] : i32
! CHECK-NEXT: }

! Check mif.alloc_coarray is still in __mif_save_coarrays_allocate for modules variables for example
! CHECK-LABEL: func.func @__mif_save_coarrays_allocate() {
! CHECK-NEXT:  %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
! CHECK-NEXT:  %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
! CHECK-NEXT:  %[[VAL_2:.*]] = mif.init -> i32
! CHECK-NEXT:  %[[VAL_3:.*]] = fir.address_of(@_QMmEcm) : !fir.ref<i32>
! CHECK-NEXT:  %[[C1:.*]] = arith.constant 1 : i64
! CHECK-NEXT:  %[[C1_I64_1:.*]] = arith.constant 1 : i64
! CHECK-NEXT:  %[[C0:.*]] = arith.constant 0 : index
! CHECK-NEXT:  %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! CHECK-NEXT:  fir.store %[[C1_I64_1]] to %[[VAL_4]] : !fir.ref<i64>
! CHECK-NEXT:  %[[VAL_5:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! CHECK-NEXT:  %[[C1_I64_2:.*]] = arith.constant 1 : i64
! CHECK-NEXT:  %[[VAL_6:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! CHECK-NEXT:  mif.alloc_coarray %[[VAL_3]] lcobounds %[[VAL_5]] ucobounds %[[VAL_6]] uniq_name("_QMmEcm") : (!fir.ref<i32>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
! CHECK-NEXT:  return
! CHECK-NEXT: }
! CHECK: fir.global internal @_QFEc : i32 {
! CHECK-NEXT:   %[[VAL_0:.*]] = fir.zero_bits i32
! CHECK-NEXT:   fir.has_value %[[VAL_0]] : i32
! CHECK-NEXT: }

