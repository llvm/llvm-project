! Test the new RANK clause.  This uses the examples from the F2023 Standard and
! related explanation documents. This test verifies that RANK clause correctly
! sets the rank of variables in HLFIR lowering output.
!
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QQmain() attributes {fir.bindc_name = "RANK_CLAUSE02"}
program rank_clause02
    implicit none

! CHECK-DAG: %{{.*}} = fir.address_of(@_QFEx0) : !fir.ref<!fir.array<10x10x10x!fir.logical<4>>>
! CHECK-DAG: %{{.*}} = fir.address_of(@_QFEarray1) : !fir.ref<!fir.array<10x10xi32>>
    logical :: X0(10,10,10)
    integer :: array1(10,10)

    interface
      subroutine sub02(arg1)
        integer, rank(2) :: arg1
      end subroutine
    end interface

    call sub01(X0)

    call sub02(array1)

  contains

! CHECK-LABEL: func.func private @_QFPsub01(
! CHECK-SAME: %[[X3_ALLOC:.*]]: !fir.box<!fir.array<?x?x?x!fir.logical<4>>>{{.*}})
    subroutine sub01(X3)

! CHECK: %[[X0_ALLOC:.*]] = fir.alloca !fir.array<10x10x10xi32>
! CHECK: %[[X0_DECL:.*]]:2 = hlfir.declare %[[X0_ALLOC]]
      integer :: X0(10,10,10)

! CHECK: %[[X1_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?x?x!fir.logical<4>>>>
! CHECK: %[[X1_DECL:.*]]:2 = hlfir.declare %[[X1_ALLOC]] {{{.*}}fortran_attrs = #fir.var_attrs<allocatable>{{.*}}} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x!fir.logical<4>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x!fir.logical<4>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x!fir.logical<4>>>>>)
      logical, rank(rank(X0)), allocatable :: X1 ! Rank 3, deferred shape

! CHECK: %[[X2_ALLOC:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xcomplex<f32>>>>
! CHECK: %[[X2_DECL:.*]]:2 = hlfir.declare %[[X2_ALLOC]] {{{.*}}fortran_attrs = #fir.var_attrs<pointer>{{.*}}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xcomplex<f32>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xcomplex<f32>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xcomplex<f32>>>>>)
      complex, rank(2), pointer :: X2 ! Rank 2, deferred-shape

! CHECK: %[[X3_DECL:.*]]:2 = hlfir.declare %[[X3_ALLOC]] dummy_scope %{{.*}} arg 1 {{.*}}: (!fir.box<!fir.array<?x?x?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?x!fir.logical<4>>>, !fir.box<!fir.array<?x?x?x!fir.logical<4>>>)
      logical, rank(rank(X0)) :: X3 ! Rank 3, assumed-shape

! CHECK: %[[X4_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<f32>>
! CHECK: %[[X4_DECL:.*]]:2 = hlfir.declare %[[X4_ALLOC]] {{{.*}}fortran_attrs = #fir.var_attrs<allocatable>{{.*}}} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
      real, rank(0) :: X4 ! Scalar
      allocatable :: X4

    end subroutine

end program

! CHECK-LABEL: func.func @_QPsub02(
! CHECK-SAME: %[[A_ALLOC:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}})
subroutine sub02(A)
! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A_ALLOC]] dummy_scope %{{.*}} arg 1 {{.*}}: (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
    integer, rank(2) :: A

! CHECK: %[[B_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B_ALLOC]] {{{.*}}fortran_attrs = #fir.var_attrs<allocatable>{{.*}}} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
    integer, allocatable, rank(2) :: B

end subroutine
