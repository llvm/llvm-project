! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test assumed shape dummy argument on callee side

! CHECK-LABEL: func @_QPtest_assumed_shape_1(%arg0: !fir.box<!fir.array<?xi32>>) 
subroutine test_assumed_shape_1(x)
  integer :: x(:)
  ! CHECK: %[[addr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  ! CHECK: %[[c0:.*]] = constant 0 : index
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %arg0, %[[c0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)

  print *, x
  ! Test extent/lower bound use in the IO statement
  ! CHECK: %[[cookie:.*]] = fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: %[[shape:.*]] = fir.shape_shift %[[dims]]#1, %[[dims]]#2 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[newbox:.*]] = fir.embox %[[addr]](%[[shape]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
  ! CHECK: %[[castedBox:.*]] = fir.convert %[[newbox]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%[[cookie]], %[[castedBox]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
end subroutine

! lower bounds all ones
! CHECK-LABEL:  func @_QPtest_assumed_shape_2(%arg0: !fir.box<!fir.array<?x?xf32>>)
subroutine test_assumed_shape_2(x)
  real :: x(1:, 1:)
  ! CHECK: fir.box_addr
  ! CHECK: %[[dims1:.*]]:3 = fir.box_dims
  ! CHECK: %[[dims2:.*]]:3 = fir.box_dims
  print *, x
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: fir.shape %[[dims1]]#2, %[[dims2]]#2
end subroutine

! explicit lower bounds different from 1
! CHECK-LABEL: func @_QPtest_assumed_shape_3(%arg0: !fir.box<!fir.array<?x?x?xi32>>)
subroutine test_assumed_shape_3(x)
  integer :: x(2:, 3:, 42:)
  ! CHECK: fir.box_addr
  ! CHECK: fir.box_dim
  ! CHECK: %[[c2_i64:.*]] = constant 2 : i64
  ! CHECK: %[[c2:.*]] = fir.convert %[[c2_i64]] : (i64) -> index
  ! CHECK: fir.box_dim
  ! CHECK: %[[c3_i64:.*]] = constant 3 : i64
  ! CHECK: %[[c3:.*]] = fir.convert %[[c3_i64]] : (i64) -> index
  ! CHECK: fir.box_dim
  ! CHECK: %[[c42_i64:.*]] = constant 42 : i64
  ! CHECK: %[[c42:.*]] = fir.convert %[[c42_i64]] : (i64) -> index

  print *, x
  ! CHECK: fir.shape_shift %[[c2]], %{{.*}}, %[[c3]], %{{.*}}, %[[c42]], %{{.*}} :
end subroutine

! Constant length
! func @_QPtest_assumed_shape_char(%arg0: !fir.box<!fir.array<10x?x!fir.char<1>>>)
subroutine test_assumed_shape_char(c)
  character(10) :: c(:)
  ! CHECK: %[[addr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<10x?x!fir.char<1>>>) -> !fir.ref<!fir.array<10x?x!fir.char<1>>>

  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<10x?x!fir.char<1>>>, index) -> (index, index, index)

  print *, c
  ! CHECK: %[[shape:.*]] = fir.shape_shift %[[dims]]#1, %[[dims]]#2 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: fir.embox %[[addr]](%[[shape]]) : (!fir.ref<!fir.array<10x?x!fir.char<1>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<10x?x!fir.char<1>>>
end subroutine

! Assumed length
! CHECK-LABEL: func @_QPtest_assumed_shape_char_2(%arg0: !fir.box<!fir.array<?x?x!fir.char<1>>>)
subroutine test_assumed_shape_char_2(c)
  character(*) :: c(:)
  ! CHECK: %[[addr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?x?x!fir.char<1>>>) -> !fir.ref<!fir.array<?x?x!fir.char<1>>>
  ! CHECK: %[[len:.*]] = fir.box_elesize %arg0 : (!fir.box<!fir.array<?x?x!fir.char<1>>>) -> index

  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?x?x!fir.char<1>>>, index) -> (index, index, index)

  print *, c
  ! CHECK: %[[shape:.*]] = fir.shape_shift %[[dims]]#1, %[[dims]]#2 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: fir.embox %[[addr]](%[[shape]]) typeparams %[[len]] : (!fir.ref<!fir.array<?x?x!fir.char<1>>>, !fir.shapeshift<1>, index) -> !fir.box<!fir.array<?x?x!fir.char<1>>>
end subroutine


! lower bounds all 1.
! CHECK: func @_QPtest_assumed_shape_char_3(%arg0: !fir.box<!fir.array<?x?x?x!fir.char<1>>>)
subroutine test_assumed_shape_char_3(c)
  character(*) :: c(1:, 1:)
  ! CHECK: fir.box_addr
  ! CHECK: fir.box_elesize
  ! CHECK: %[[dims1:.*]]:3 = fir.box_dims
  ! CHECK: %[[dims2:.*]]:3 = fir.box_dims
  print *, c
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: fir.shape %[[dims1]]#2, %[[dims2]]#2
end subroutine
