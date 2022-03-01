! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test allocatable dummy argument on callee side

! CHECK-LABEL: func @_QPtest_scalar(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>{{.*}})
subroutine test_scalar(x)
  real, allocatable :: x

  print *, x
  ! CHECK: %[[box:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! CHECK: %[[val:.*]] = fir.load %[[addr]] : !fir.heap<f32>
end subroutine

! CHECK-LABEL: func @_QPtest_array(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>{{.*}})
subroutine test_array(x)
  integer, allocatable :: x(:,:)

  print *, x(1,2)
  ! CHECK: %[[box:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK-DAG: fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
  ! CHECK-DAG: fir.box_dims %[[box]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>, index) -> (index, index, index)
  ! CHECK-DAG: fir.box_dims %[[box]], %c1{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>, index) -> (index, index, index)
end subroutine

! CHECK-LABEL: func @_QPtest_char_scalar_deferred(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>{{.*}})
subroutine test_char_scalar_deferred(c)
  character(:), allocatable :: c
  external foo1
  call foo1(c)
  ! CHECK: %[[box:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
  ! CHECK-DAG: %[[addr_cast:.*]] = fir.convert %[[addr]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr_cast]], %[[len]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPfoo1(%[[boxchar]]) : (!fir.boxchar<1>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_char_scalar_explicit_cst(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>{{.*}})
subroutine test_char_scalar_explicit_cst(c)
  character(10), allocatable :: c
  external foo1
  call foo1(c)
  ! CHECK: %[[box:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.char<1,10>>>) -> !fir.heap<!fir.char<1,10>>
  ! CHECK-DAG: %[[addr_cast:.*]] = fir.convert %[[addr]] : (!fir.heap<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr_cast]], %c10{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPfoo1(%[[boxchar]]) : (!fir.boxchar<1>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_char_scalar_explicit_dynamic(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}})
subroutine test_char_scalar_explicit_dynamic(c, n)
  integer :: n
  character(n), allocatable :: c
  external foo1
  ! Check that the length expr was evaluated before the execution parts.
  ! CHECK: %[[len:.*]] = fir.load %arg1 : !fir.ref<i32>
  n = n + 1
  ! CHECK: fir.store {{.*}} to %arg1 : !fir.ref<i32>
  call foo1(c)
  ! CHECK: %[[box:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  ! CHECK-DAG: %[[addr_cast:.*]] = fir.convert %[[addr]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK-DAG: %[[len_cast:.*]] = fir.convert %[[len]] : (i32) -> index
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr_cast]], %[[len_cast]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPfoo1(%[[boxchar]]) : (!fir.boxchar<1>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_char_array_deferred(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>{{.*}})
subroutine test_char_array_deferred(c)
  character(:), allocatable :: c(:)
  external foo1
  call foo1(c(10))
  ! CHECK: %[[box:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
  ! CHECK-DAG: fir.box_dims %[[box]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> index
  ! [...] address computation
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %{{.*}}, %[[len]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPfoo1(%[[boxchar]]) : (!fir.boxchar<1>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_char_array_explicit_cst(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>{{.*}})
subroutine test_char_array_explicit_cst(c)
  character(10), allocatable :: c(:)
  external foo1
  call foo1(c(3))
  ! CHECK: %[[box:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,10>>>
  ! [...] address computation
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %{{.*}}, %c10{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPfoo1(%[[boxchar]]) : (!fir.boxchar<1>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_char_array_explicit_dynamic(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}})
subroutine test_char_array_explicit_dynamic(c, n)
  integer :: n
  character(n), allocatable :: c(:)
  external foo1
  ! Check that the length expr was evaluated before the execution parts.
  ! CHECK: %[[len:.*]] = fir.load %arg1 : !fir.ref<i32>
  n = n + 1
  ! CHECK: fir.store {{.*}} to %arg1 : !fir.ref<i32>
  call foo1(c(1))
  ! CHECK: %[[box:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
  ! [...] address computation
  ! CHECK: fir.coordinate_of
  ! CHECK-DAG: %[[len_cast:.*]] = fir.convert %[[len]] : (i32) -> index
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %{{.*}}, %[[len_cast]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPfoo1(%[[boxchar]]) : (!fir.boxchar<1>) -> ()
end subroutine

! Check that when reading allocatable length from descriptor, the width is taking
! into account when the kind is not 1.

! CHECK-LABEL: func @_QPtest_char_scalar_deferred_k2(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<2,?>>>>{{.*}})
subroutine test_char_scalar_deferred_k2(c)
  character(kind=2, len=:), allocatable :: c
  external foo2
  call foo2(c)
  ! CHECK: %[[box:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<2,?>>>>
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.char<2,?>>>) -> !fir.heap<!fir.char<2,?>>
  ! CHECK-DAG: %[[size:.*]] = fir.box_elesize %[[box]] : (!fir.box<!fir.heap<!fir.char<2,?>>>) -> index
  ! CHECK-DAG: %[[len:.*]] = arith.divsi %[[size]], %c2{{.*}} : index
  ! CHECK-DAG: %[[addr_cast:.*]] = fir.convert %[[addr]] : (!fir.heap<!fir.char<2,?>>) -> !fir.ref<!fir.char<2,?>>
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr_cast]], %[[len]] : (!fir.ref<!fir.char<2,?>>, index) -> !fir.boxchar<2>
  ! CHECK: fir.call @_QPfoo2(%[[boxchar]]) : (!fir.boxchar<2>) -> ()
end subroutine
