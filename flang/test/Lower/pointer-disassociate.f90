! Test lowering of pointer disassociation
! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s


! -----------------------------------------------------------------------------
!     Test p => NULL()
! -----------------------------------------------------------------------------


! CHECK-LABEL: func @_QPtest_scalar(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>{{.*}})
subroutine test_scalar(p)
  real, pointer :: p
  ! CHECK: %[[null:.*]] = fir.zero_bits !fir.ptr<f32>
  ! CHECK: %[[box:.*]] = fir.embox %[[null]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  p => NULL()
end subroutine

! CHECK-LABEL: func @_QPtest_scalar_char(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>{{.*}})
subroutine test_scalar_char(p)
  character(:), pointer :: p
  ! CHECK: %[[null:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[box:.*]] = fir.embox %[[null]] typeparams %c0{{.*}} : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  p => NULL()
end subroutine

! CHECK-LABEL: func @_QPtest_array(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}})
subroutine test_array(p)
  real, pointer :: p(:)
  ! CHECK: %[[null:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[shape:.*]] = fir.shape %c0{{.*}}
  ! CHECK: %[[box:.*]] = fir.embox %[[null]](%[[shape]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => NULL()
end subroutine

! Test p(lb, ub) => NULL() which is none sens but is not illegal.
! CHECK-LABEL: func @_QPtest_array_remap(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}})
subroutine test_array_remap(p)
  real, pointer :: p(:)
  ! CHECK: %[[null:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[shape:.*]] = fir.shape %c0{{.*}}
  ! CHECK: %[[box:.*]] = fir.embox %[[null]](%[[shape]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p(10:20) => NULL()
end subroutine

! -----------------------------------------------------------------------------
!     Test p => NULL(MOLD)
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest_scalar_mold(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>{{[^,]*}},
subroutine test_scalar_mold(p, x)
  real, pointer :: p, x
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<f32>>
  ! CHECK: %[[VAL_1:.*]] = fir.zero_bits !fir.ptr<f32>
  ! CHECK: %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  ! CHECK: %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  ! CHECK: %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  ! CHECK: %[[VAL_5:.*]] = fir.embox %[[VAL_4]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.store %[[VAL_5]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  p => NULL(x)
end subroutine

! CHECK-LABEL: func @_QPtest_scalar_char_mold(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>{{[^,]*}},
subroutine test_scalar_char_mold(p, x)
  character(:), pointer :: p, x
  ! CHECK: %[[VAL_7:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: %[[VAL_8:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[VAL_9:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_10:.*]] = fir.embox %[[VAL_8]] typeparams %[[VAL_9]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[VAL_10]] to %[[VAL_7]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  ! CHECK: %[[VAL_11:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  ! CHECK: %[[VAL_12:.*]] = fir.box_elesize %[[VAL_11]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
  ! CHECK: %[[VAL_13:.*]] = fir.box_addr %[[VAL_11]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[VAL_14:.*]] = fir.embox %[[VAL_13]] typeparams %[[VAL_12]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[VAL_14]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  p => NULL(x)
end subroutine

! CHECK-LABEL: func @_QPtest_array_mold(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{[^,]*}},
subroutine test_array_mold(p, x)
  real, pointer :: p(:), x(:)
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: %[[VAL_1:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[VAL_2:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_4:.*]] = fir.embox %[[VAL_1]](%[[VAL_3]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[VAL_4]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[VAL_6:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK: %[[VAL_8:.*]] = fir.shift %[[VAL_7]]#0 : (index) -> !fir.shift<1>
  ! CHECK: %[[VAL_9:.*]] = fir.rebox %[[VAL_5]](%[[VAL_8]]) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[VAL_9]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => NULL(x)
end subroutine

subroutine test_polymorphic_null(p)
  type t
  end type
  class(t), pointer :: p(:)
  p => null()
end subroutine
! CHECK-LABEL:   func.func @_QPtest_polymorphic_null(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFtest_polymorphic_nullTt>>>>>
! CHECK:  %[[VAL_1:.*]] = fir.type_desc !fir.type<_QFtest_polymorphic_nullTt> 
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFtest_polymorphic_nullTt>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (!fir.tdesc<!fir.type<_QFtest_polymorphic_nullTt>>) -> !fir.ref<none> 
! CHECK:  %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_6:.*]] = fir.call @_FortranAPointerNullifyDerived(%[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none

subroutine test_unlimited_polymorphic_null(p)
  class(*), pointer :: p(:)
  p => null()
end subroutine
! CHECK-LABEL:   func.func @_QPtest_unlimited_polymorphic_null(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK:  %[[VAL_1:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xnone>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_4:.*]] = fir.embox %[[VAL_1]](%[[VAL_3]]) : (!fir.ptr<!fir.array<?xnone>>, !fir.shape<1>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
! CHECK:  fir.store %[[VAL_4]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
