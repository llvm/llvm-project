! Test that EQUIVALENCE'd variables with lastprivate are correctly privatized.
! EQUIVALENCE aliases use fir.ptr (via castAliasToPointer), but are not true
! Fortran POINTERs. The privatizer must unwrap fir.ptr and allocate the
! underlying data type.

!RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! Privatizers appear at module scope; second subroutine's privatizer comes first.
!CHECK: omp.private {type = private} @[[C1_PRIV:.*Ec1_private.*]] : !fir.char<1,10>

! The array privatizer operates on the unwrapped fir.ptr data type. A constant-
! shape, trivial-element array privatizes unboxed as a plain fir.array (no
! descriptor, no init/dealloc region). The end-of-line anchor confirms there is
! no `init {` region.
!CHECK: omp.private {type = private} @[[A_PRIV:.*Ea_private.*]] : !fir.array<10xi32>{{$}}

!CHECK-LABEL: func.func @_QPlastprivate_equivalence()
!CHECK: %[[AGG:.*]] = fir.alloca !fir.array<40xi8>
!CHECK: %[[A_PTR:.*]] = fir.convert %{{.*}} : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xi32>>
!CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A_PTR]](%{{.*}}) storage(%[[AGG]][0]) {uniq_name = "_QFlastprivate_equivalenceEa"}
!CHECK: omp.parallel {
!CHECK:   omp.wsloop private(@[[A_PRIV]] %{{.*}}
!CHECK:     omp.loop_nest
! Verify lastprivate writeback copies the private array to the original
! EQUIVALENCE alias address.
!CHECK:       %[[A_PRIV_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}Ea"}
!CHECK:       fir.if %{{.*}} {
!CHECK:         hlfir.assign %[[A_PRIV_DECL]]#0 to %[[A_DECL]]#0 : !fir.ptr<!fir.array<10xi32>>, !fir.ptr<!fir.array<10xi32>>
!CHECK:       }
!CHECK:       omp.yield
!CHECK:     }
!CHECK:   }
!CHECK:   omp.terminator
!CHECK: }
subroutine lastprivate_equivalence
  integer :: a(10), b(1)
  equivalence (a(1), b(1))
  !$omp parallel do lastprivate(a)
  do i = 1, 10
    a(:) = i
  end do
  !$omp end parallel do
end subroutine

!CHECK-LABEL: func.func @_QPlastprivate_equiv_char()
!CHECK: %[[CAGG:.*]] = fir.alloca !fir.array<10xi8>
!CHECK: %[[C1_PTR:.*]] = fir.convert %{{.*}} : (!fir.ref<i8>) -> !fir.ptr<!fir.char<1,10>>
!CHECK: %[[C1_DECL:.*]]:2 = hlfir.declare %[[C1_PTR]] typeparams %{{.*}} storage(%[[CAGG]][0]) {uniq_name = "_QFlastprivate_equiv_charEc1"}
!CHECK: omp.parallel {
!CHECK:   omp.wsloop private(@[[C1_PRIV]] %{{.*}}
!CHECK:     omp.loop_nest
!CHECK:       fir.if %{{.*}} {
!CHECK:         hlfir.assign %{{.*}} to %[[C1_DECL]]#0 : !fir.ptr<!fir.char<1,10>>, !fir.ptr<!fir.char<1,10>>
!CHECK:       }
!CHECK:       omp.yield
!CHECK:     }
!CHECK:   }
!CHECK:   omp.terminator
!CHECK: }
subroutine lastprivate_equiv_char
  character(10) :: c1, c2
  equivalence (c1, c2)
  !$omp parallel do lastprivate(c1)
  do i = 1, 4
    c1 = 'test'
  end do
  !$omp end parallel do
end subroutine
