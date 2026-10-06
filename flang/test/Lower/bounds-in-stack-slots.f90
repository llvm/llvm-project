! Test that at -O0 the non-constant bounds and length parameters of variables
! are stored to stack slots, for the debug info.
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -O1 %s -o - | FileCheck %s --check-prefix=OPT
! RUN: %flang_fc1 -emit-hlfir -fopenmp -O2 %s -o - | FileCheck %s --check-prefix=OPT
! RUN: %flang_fc1 -emit-hlfir -fopenmp -O3 %s -o - | FileCheck %s --check-prefix=OPT
! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir -mlir-print-debuginfo %s -o - \
! RUN:   | FileCheck %s --check-prefix=LOC

! OPT-NOT: fir.debug_bound_slot

! The slot and the store are not anything the user wrote, so they stay out of
! the line table.
! LOC-LABEL: func.func @_QPchar_len(
! LOC:         %[[SLOT:.*]] = fir.alloca index {fir.debug_bound_slot} loc(#[[LOC:loc[0-9]+]])
! LOC:         fir.store %{{.*}} to %[[SLOT]] : !fir.ref<index> loc(#[[LOC]])
! LOC:       #[[LOC]] = loc(unknown)

! The constant extent does not need a slot.
! CHECK-LABEL: func.func @_QPexplicit_shape(
! CHECK:         %[[SLOT:.*]] = fir.alloca index {fir.debug_bound_slot}
! CHECK-NOT:     fir.debug_bound_slot
! CHECK:         %[[SHAPE:.*]] = fir.shape %{{.*}}, %[[EXTENT:.*]] : (index, index) -> !fir.shape<2>
! CHECK-NEXT:    fir.store %[[EXTENT]] to %[[SLOT]] : !fir.ref<index>
! CHECK-NEXT:    hlfir.declare %{{.*}}(%[[SHAPE]]) {{.*}}uniq_name("_QFexplicit_shapeEa")
subroutine explicit_shape(a, n)
  integer :: n
  real :: a(4, n)
end subroutine

! CHECK-LABEL: func.func @_QPchar_len(
! CHECK:         %[[SLOT:.*]] = fir.alloca index {fir.debug_bound_slot}
! CHECK:         %[[UNBOX:.*]]:2 = fir.unboxchar
! CHECK-NEXT:    fir.store %[[UNBOX]]#1 to %[[SLOT]] : !fir.ref<index>
! CHECK-NEXT:    hlfir.declare %[[UNBOX]]#0 typeparams %[[UNBOX]]#1 {{.*}}uniq_name("_QFchar_lenEstr")
subroutine char_len(str)
  character(*) :: str
end subroutine

! The assumed-size sentinel gets no slot.
! CHECK-LABEL: func.func @_QPassumed_size(
! CHECK-NOT:     fir.debug_bound_slot
! CHECK:         hlfir.declare %{{.*}}uniq_name("_QFassumed_sizeEa")
subroutine assumed_size(a)
  real :: a(*)
  a(1) = 0.0
end subroutine

! CHECK-LABEL: func.func @_QPassumed_shape(
! CHECK:         %[[SLOT:.*]] = fir.alloca index {fir.debug_bound_slot}
! CHECK:         %[[SHIFT:.*]] = fir.shift %[[LB:.*]] : (index) -> !fir.shift<1>
! CHECK-NEXT:    fir.store %[[LB]] to %[[SLOT]] : !fir.ref<index>
! CHECK-NEXT:    hlfir.declare %{{.*}}(%[[SHIFT]]) {{.*}}uniq_name("_QFassumed_shapeEx")
subroutine assumed_shape(x, n)
  integer :: n
  real :: x(n:)
end subroutine

! The slot is in the entry block of the function.
! CHECK-LABEL: func.func @_QPblock_construct(
! CHECK:         %[[SLOT:.*]] = fir.alloca index {fir.debug_bound_slot}
! CHECK:         %[[SHAPE:.*]] = fir.shape %[[EXTENT:.*]] : (index) -> !fir.shape<1>
! CHECK-NEXT:    fir.store %[[EXTENT]] to %[[SLOT]] : !fir.ref<index>
! CHECK-NEXT:    hlfir.declare %{{.*}}(%[[SHAPE]]) uniq_name("_QFblock_constructB1Eb")
subroutine block_construct(n)
  integer :: n
  block
    real :: b(n)
    b = 0.0
  end block
end subroutine

! The declaration in the target region has slots of its own.
! CHECK-LABEL: func.func @_QPtarget_region(
! CHECK:         omp.target
! CHECK:           %[[COUNT_SLOT:.*]] = fir.alloca index <{pinned}> {fir.debug_bound_slot}
! CHECK:           %[[LB_SLOT:.*]] = fir.alloca index <{pinned}> {fir.debug_bound_slot}
! CHECK:           %[[SHAPE:.*]] = fir.shape_shift %[[LB:.*]], %[[EXTENT:.*]] : (index, index) -> !fir.shapeshift<1>
! CHECK-NEXT:      fir.store %[[LB]] to %[[LB_SLOT]] : !fir.ref<index>
! CHECK-NEXT:      fir.store %[[EXTENT]] to %[[COUNT_SLOT]] : !fir.ref<index>
! CHECK-NEXT:      hlfir.declare %{{.*}}(%[[SHAPE]]) uniq_name("_QFtarget_regionEarr")
subroutine target_region(lb, ub)
  integer :: lb, ub
  real :: arr(lb:ub)
  !$omp target map(tofrom: arr)
  arr = 1.0
  !$omp end target
end subroutine
