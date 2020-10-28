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
  ! CHECK-DAG: %[[c42:.*]] = constant 42 : index
  ! CHECK-DAG: %[[c55:.*]] = constant 55 : index
  ! CHECK-DAG: %[[c12:.*]] = constant 12 : index
  ! CHECK-DAG: %[[addr:.*]] = fir.alloca !fir.array<42x55x12xf32> {name = "_QFfooEx"}

  call bar(x)
  ! CHECK: %[[shape:.*]] = fir.shape %[[c42]], %[[c55]], %[[c12]] : (index, index, index) -> !fir.shape<3>
  ! CHECK: %[[embox:.*]] = fir.embox %[[addr]](%[[shape]]) : (!fir.ref<!fir.array<42x55x12xf32>>, !fir.shape<3>) -> !fir.box<!fir.array<42x55x12xf32>>
  ! CHECK: %[[castedBox:.*]] = fir.convert %[[embox]] : (!fir.box<!fir.array<42x55x12xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
  ! CHECK: fir.call @_QPbar(%[[castedBox]]) : (!fir.box<!fir.array<?x?x?xf32>>) -> ()
end subroutine


! Test passing character array as assumed shape.
! CHECK-LABEL: func @_QPfoo_char(%arg0: !fir.boxchar<1>)
subroutine foo_char(x)
  interface
    subroutine bar_char(x)
      character(*) :: x(1:, 10:, :)
    end subroutine
  end interface
  character(*) :: x(42, 55, 12)
  ! CHECK-DAG: %[[x:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, index)
  ! CHECK-DAG: %[[addr:.*]] = fir.convert %[[x]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.array<?x42x55x12x!fir.char<1>>>
  ! CHECK-DAG: %[[c42:.*]] = constant 42 : index
  ! CHECK-DAG: %[[c55:.*]] = constant 55 : index
  ! CHECK-DAG: %[[c12:.*]] = constant 12 : index

  call bar_char(x)
  ! CHECK: %[[shape:.*]] = fir.shape %[[c42]], %[[c55]], %[[c12]] : (index, index, index) -> !fir.shape<3>
  ! CHECK: %[[embox:.*]] = fir.embox %[[addr]](%[[shape]]) typeparams %[[x]]#1 : (!fir.ref<!fir.array<?x42x55x12x!fir.char<1>>>, !fir.shape<3>, index) -> !fir.box<!fir.array<?x42x55x12x!fir.char<1>>>
  ! CHECK: %[[castedBox:.*]] = fir.convert %[[embox]] : (!fir.box<!fir.array<?x42x55x12x!fir.char<1>>>) -> !fir.box<!fir.array<?x?x?x?x!fir.char<1>>>
  ! CHECK: fir.call @_QPbar_char(%[[castedBox]]) : (!fir.box<!fir.array<?x?x?x?x!fir.char<1>>>) -> ()
end subroutine

! Test external function declarations

! CHECK: func @_QPbar(!fir.box<!fir.array<?x?x?xf32>>)
! CHECK: func @_QPbar_char(!fir.box<!fir.array<?x?x?x?x!fir.char<1>>>)
