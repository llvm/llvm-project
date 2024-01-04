! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! Test assumed shape dummy argument on callee side

! TODO: These tests rely on looking at how a new fir.box is made for an assumed shape
! to see if lowering lowered and mapped the assumed shape symbol properties.
! However, the argument fir.box of the assumed shape could also be used instead
! of making a new fir.box and this would break all these tests. In fact, for non
! contiguous arrays, this is the case. Find a better way to tests symbol lowering/mapping.

! CHECK-LABEL: func @_QPtest_assumed_shape_1(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x", fir.contiguous}) 
subroutine test_assumed_shape_1(x)
  integer, contiguous :: x(:)
  ! CHECK: %[[addr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  ! CHECK: %[[c0:.*]] = arith.constant 0 : index
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %arg0, %[[c0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  ! CHECK: %[[c1:.*]] = arith.constant 1 : index

  print *, x
  ! Test extent/lower bound use in the IO statement
  ! CHECK: %[[cookie:.*]] = fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: %[[shape:.*]] = fir.shape_shift %[[c1]], %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[newbox:.*]] = fir.embox %[[addr]](%[[shape]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
  ! CHECK: %[[castedBox:.*]] = fir.convert %[[newbox]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%[[cookie]], %[[castedBox]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
end subroutine

! lower bounds all ones
! CHECK-LABEL:  func @_QPtest_assumed_shape_2(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x", fir.contiguous})
subroutine test_assumed_shape_2(x)
  real, contiguous :: x(1:, 1:)
  ! CHECK: fir.box_addr
  ! CHECK: %[[dims1:.*]]:3 = fir.box_dims
  ! CHECK: %[[dims2:.*]]:3 = fir.box_dims
  print *, x
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: fir.shape %[[dims1]]#1, %[[dims2]]#1
end subroutine

! explicit lower bounds different from 1
! CHECK-LABEL: func @_QPtest_assumed_shape_3(%arg0: !fir.box<!fir.array<?x?x?xi32>> {fir.bindc_name = "x", fir.contiguous})
subroutine test_assumed_shape_3(x)
  integer, contiguous :: x(2:, 3:, 42:)
  ! CHECK: fir.box_addr
  ! CHECK: fir.box_dim
  ! CHECK: %[[c2_i64:.*]] = arith.constant 2 : i64
  ! CHECK: %[[c2:.*]] = fir.convert %[[c2_i64]] : (i64) -> index
  ! CHECK: fir.box_dim
  ! CHECK: %[[c3_i64:.*]] = arith.constant 3 : i64
  ! CHECK: %[[c3:.*]] = fir.convert %[[c3_i64]] : (i64) -> index
  ! CHECK: fir.box_dim
  ! CHECK: %[[c42_i64:.*]] = arith.constant 42 : i64
  ! CHECK: %[[c42:.*]] = fir.convert %[[c42_i64]] : (i64) -> index

  print *, x
  ! CHECK: fir.shape_shift %[[c2]], %{{.*}}, %[[c3]], %{{.*}}, %[[c42]], %{{.*}} :
end subroutine

! Constant length
! func @_QPtest_assumed_shape_char(%arg0: !fir.box<!fir.array<?x!fir.char<1,10>>> {fir.bindc_name = "c", fir.contiguous})
subroutine test_assumed_shape_char(c)
  character(10), contiguous :: c(:)
  ! CHECK: %[[addr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> !fir.ref<!fir.array<?x!fir.char<1,10>>>

  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?x!fir.char<1,10>>>, index) -> (index, index, index)
  ! CHECK: %[[c1:.*]] = arith.constant 1 : index

  print *, c
  ! CHECK: %[[shape:.*]] = fir.shape_shift %[[c1]], %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: fir.embox %[[addr]](%[[shape]]) : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.char<1,10>>>
end subroutine

! Assumed length
! CHECK-LABEL: func @_QPtest_assumed_shape_char_2(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c", fir.contiguous})
subroutine test_assumed_shape_char_2(c)
  character(*), contiguous :: c(:)
  ! CHECK: %[[addr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[len:.*]] = fir.box_elesize %arg0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index

  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
  ! CHECK: %[[c1:.*]] = arith.constant 1 : index

  print *, c
  ! CHECK: %[[shape:.*]] = fir.shape_shift %[[c1]], %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: fir.embox %[[addr]](%[[shape]]) typeparams %[[len]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shapeshift<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
end subroutine


! lower bounds all 1.
! CHECK: func @_QPtest_assumed_shape_char_3(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,?>>> {fir.bindc_name = "c", fir.contiguous})
subroutine test_assumed_shape_char_3(c)
  character(*), contiguous :: c(1:, 1:)
  ! CHECK: fir.box_addr
  ! CHECK: fir.box_elesize
  ! CHECK: %[[dims1:.*]]:3 = fir.box_dims
  ! CHECK: %[[dims2:.*]]:3 = fir.box_dims
  print *, c
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: fir.shape %[[dims1]]#1, %[[dims2]]#1
end subroutine
