! RUN: bbc -emit-fir %s -o - | FileCheck %s

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

  call bar(x)
  ! CHECK: %[[shape:.*]] = fir.shape %[[c42]], %[[c55]], %[[c12]] : (index, index, index) -> !fir.shape<3>
  ! CHECK: %[[embox:.*]] = fir.embox %[[addr]](%[[shape]]) : (!fir.ref<!fir.array<42x55x12xf32>>, !fir.shape<3>) -> !fir.box<!fir.array<42x55x12xf32>>
  ! CHECK: %[[castedBox:.*]] = fir.convert %[[embox]] : (!fir.box<!fir.array<42x55x12xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
  ! CHECK: fir.call @_QPbar(%[[castedBox]]) : (!fir.box<!fir.array<?x?x?xf32>>) -> ()
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
  ! CHECK-DAG: %[[x:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-DAG: %[[addr:.*]] = fir.convert %[[x]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<42x55x12x!fir.char<1,?>>>
  ! CHECK-DAG: %[[c42:.*]] = arith.constant 42 : index
  ! CHECK-DAG: %[[c55:.*]] = arith.constant 55 : index
  ! CHECK-DAG: %[[c12:.*]] = arith.constant 12 : index

  call bar_char(x)
  ! CHECK: %[[shape:.*]] = fir.shape %[[c42]], %[[c55]], %[[c12]] : (index, index, index) -> !fir.shape<3>
  ! CHECK: %[[embox:.*]] = fir.embox %[[addr]](%[[shape]]) typeparams %[[x]]#1 : (!fir.ref<!fir.array<42x55x12x!fir.char<1,?>>>, !fir.shape<3>, index) -> !fir.box<!fir.array<42x55x12x!fir.char<1,?>>>
  ! CHECK: %[[castedBox:.*]] = fir.convert %[[embox]] : (!fir.box<!fir.array<42x55x12x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x?x?x!fir.char<1,?>>>
  ! CHECK: fir.call @_QPbar_char(%[[castedBox]]) : (!fir.box<!fir.array<?x?x?x!fir.char<1,?>>>) -> ()
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
! CHECK:  %[[VAL_2:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_3]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_7:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:  %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_6]]#1, %[[VAL_4]]#1 : index
! CHECK:  %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_4]]#1, %[[VAL_6]]#1 : index
! CHECK:  %[[VAL_10:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
! CHECK:  %[[VAL_11:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_9]] {uniq_name = ".array.expr"}
! CHECK:  %[[VAL_12:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_13:.*]] = fir.array_load %[[VAL_11]](%[[VAL_12]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:  %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_16:.*]] = arith.subi %[[VAL_9]], %[[VAL_14]] : index
! CHECK:  %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_16]] step %[[VAL_14]] unordered iter_args(%[[VAL_19:.*]] = %[[VAL_13]]) -> (!fir.array<?xf32>) {
! CHECK:    %[[VAL_20:.*]] = fir.array_fetch %[[VAL_7]], %[[VAL_18]] : (!fir.array<?xi32>, index) -> i32
! CHECK:    %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
! CHECK:    %[[VAL_22:.*]] = arith.subi %[[VAL_21]], %[[VAL_2]] : index
! CHECK:    %[[VAL_23:.*]] = fir.array_fetch %[[VAL_10]], %[[VAL_22]] : (!fir.array<?xf32>, index) -> f32
! CHECK:    %[[VAL_24:.*]] = fir.array_update %[[VAL_19]], %[[VAL_23]], %[[VAL_18]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:    fir.result %[[VAL_24]] : !fir.array<?xf32>
! CHECK:  }
! CHECK:  fir.array_merge_store %[[VAL_13]], %[[VAL_25:.*]] to %[[VAL_11]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
! CHECK:  %[[VAL_26:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_27:.*]] = fir.embox %[[VAL_11]](%[[VAL_26]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_box(%[[VAL_27]]) : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  fir.freemem %[[VAL_11]] : !fir.heap<!fir.array<?xf32>>
end subroutine

! Test external function declarations

! CHECK: func private @_QPbar(!fir.box<!fir.array<?x?x?xf32>>)
! CHECK: func private @_QPbar_char(!fir.box<!fir.array<?x?x?x!fir.char<1,?>>>)
