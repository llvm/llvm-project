! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test passing arrays to assumed shape dummy arguments

! CHECK-LABEL: func @_QPfoo()
subroutine foo()
  interface
    subroutine bar(x)
      ! lbounds are meaningless on caller side, some are added
      ! here to check they are ignored.
      real :: x(1:, 10:, :)
    end subroutine
  end interface
  real :: x(42, 55, 12)
  ! CHECK-DAG: %[[c42:.*]] = arith.constant 42 : index
  ! CHECK-DAG: %[[c55:.*]] = arith.constant 55 : index
  ! CHECK-DAG: %[[c12:.*]] = arith.constant 12 : index
  ! CHECK-DAG: %[[addr:.*]] = fir.alloca !fir.array<42x55x12xf32> {{{.*}}uniq_name = "_QFfooEx"}
  ! CHECK: %[[shape:.*]] = fir.shape %[[c42]], %[[c55]], %[[c12]] : (index, index, index) -> !fir.shape<3>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[addr]](%[[shape]]) {uniq_name = "_QFfooEx"}

  call bar(x)
  ! CHECK: %[[embox:.*]] = fir.embox %[[x]]#0(%[[shape]]) : (!fir.ref<!fir.array<42x55x12xf32>>, !fir.shape<3>) -> !fir.box<!fir.array<42x55x12xf32>>
  ! CHECK: %[[castedBox:.*]] = fir.convert %[[embox]] : (!fir.box<!fir.array<42x55x12xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
  ! CHECK: fir.call @_QPbar(%[[castedBox]]) {{.*}}: (!fir.box<!fir.array<?x?x?xf32>>) -> ()
end subroutine


! Test passing character array as assumed shape.
! CHECK-LABEL: func @_QPfoo_char(%arg0: !fir.boxchar<1>{{.*}})
subroutine foo_char(x)
  interface
    subroutine bar_char(x)
      character(*) :: x(1:, 10:, :)
    end subroutine
  end interface
  character(*) :: x(42, 55, 12)
  ! CHECK-DAG: %[[unb:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-DAG: %[[addr:.*]] = fir.convert %[[unb]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<42x55x12x!fir.char<1,?>>>
  ! CHECK-DAG: %[[c42:.*]] = arith.constant 42 : index
  ! CHECK-DAG: %[[c55:.*]] = arith.constant 55 : index
  ! CHECK-DAG: %[[c12:.*]] = arith.constant 12 : index
  ! CHECK: %[[shape:.*]] = fir.shape %[[c42]], %[[c55]], %[[c12]] : (index, index, index) -> !fir.shape<3>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[addr]](%[[shape]]) typeparams %[[unb]]#1{{.*}} {uniq_name = "_QFfoo_charEx"}{{.*}} -> (!fir.box<!fir.array<42x55x12x!fir.char<1,?>>>, !fir.ref<!fir.array<42x55x12x!fir.char<1,?>>>)

  call bar_char(x)
  ! CHECK: %[[castedBox:.*]] = fir.convert %[[x]]#0 : (!fir.box<!fir.array<42x55x12x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x?x?x!fir.char<1,?>>>
  ! CHECK: fir.call @_QPbar_char(%[[castedBox]]) {{.*}}: (!fir.box<!fir.array<?x?x?x!fir.char<1,?>>>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_vector_subcripted_section_to_box(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "v"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
subroutine test_vector_subcripted_section_to_box(v, x)
  ! Test that a copy is made when passing a vector subscripted variable to
  ! an assumed shape argument.
  interface
    subroutine takes_box(y)
      real :: y(:)
    end subroutine
  end interface
  integer :: v(:)
  real :: x(:)
  call takes_box(x(v))
! CHECK:  %[[V:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QFtest_vector_subcripted_section_to_boxEv"}
! CHECK:  %[[X:.*]]:2 = hlfir.declare %[[VAL_1]] {{.*}} {uniq_name = "_QFtest_vector_subcripted_section_to_boxEx"}
! CHECK:  %[[c0:.*]] = arith.constant 0 : index
! CHECK:  %[[VDIMS:.*]]:3 = fir.box_dims %[[V]]#0, %[[c0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:  %[[SHAPE_V:.*]] = fir.shape %[[VDIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VEXPR:.*]] = hlfir.elemental %[[SHAPE_V]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi64> {
! CHECK:  ^bb0(%[[I:.*]]: index):
! CHECK:    %[[VELEM:.*]] = hlfir.designate %[[V]]#0 (%[[I]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VVAL:.*]] = fir.load %[[VELEM]] : !fir.ref<i32>
! CHECK:    %[[VVAL64:.*]] = fir.convert %[[VVAL]] : (i32) -> i64
! CHECK:    hlfir.yield_element %[[VVAL64]] : i64
! CHECK:  }
! CHECK:  %[[SHAPE_X:.*]] = fir.shape %[[VDIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[XEXPR:.*]] = hlfir.elemental %[[SHAPE_X]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:  ^bb0(%[[I2:.*]]: index):
! CHECK:    %[[IDX:.*]] = hlfir.apply %[[VEXPR]], %[[I2]] : (!hlfir.expr<?xi64>, index) -> i64
! CHECK:    %[[XELEM:.*]] = hlfir.designate %[[X]]#0 (%[[IDX]])  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK:    %[[XVAL:.*]] = fir.load %[[XELEM]] : !fir.ref<f32>
! CHECK:    hlfir.yield_element %[[XVAL]] : f32
! CHECK:  }
! CHECK:  %[[ASSOC:.*]]:3 = hlfir.associate %[[XEXPR]](%[[SHAPE_X]]) {{.*}}: (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_box(%[[ASSOC]]#0) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  hlfir.end_associate %[[ASSOC]]#1, %[[ASSOC]]#2 : !fir.ref<!fir.array<?xf32>>, i1
! CHECK:  hlfir.destroy %[[XEXPR]] : !hlfir.expr<?xf32>
! CHECK:  hlfir.destroy %[[VEXPR]] : !hlfir.expr<?xi64>
end subroutine

! Test external function declarations

! CHECK: func private @_QPbar(!fir.box<!fir.array<?x?x?xf32>>)
! CHECK: func private @_QPbar_char(!fir.box<!fir.array<?x?x?x!fir.char<1,?>>>)
