! Test delayed privatization for derived types with default initialization.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 |\
! RUN:   FileCheck %s

subroutine delayed_privatization_default_init
  implicit none
  type t
    integer :: i = 2
  end type
  integer :: i, res(4)
  type(t) :: a
  !$omp parallel private(a)
    call do_something(a%i)
  !$omp end parallel
end subroutine

subroutine delayed_privatization_default_init_firstprivate
  implicit none
  type t
    integer :: i = 2
  end type
  integer :: i, res(4)
  type(t) :: a
  !$omp parallel firstprivate(a)
    call do_something(a%i)
  !$omp end parallel
end subroutine

! CHECK-LABEL:   omp.private {type = firstprivate}
! CHECK-SAME:        @_QFdelayed_privatization_default_init_firstprivateEa_firstprivate_rec__QFdelayed_privatization_default_init_firstprivateTt :
! CHECK-SAME:        [[TYPE:!fir.type<_QFdelayed_privatization_default_init_firstprivateTt{i:i32}>]] copy {

! CHECK-LABEL:   omp.private {type = private}
! CHECK-SAME:        @_QFdelayed_privatization_default_initEa_private_rec__QFdelayed_privatization_default_initTt :
! CHECK-SAME:        [[TYPE:!fir.type<_QFdelayed_privatization_default_initTt{i:i32}>]] init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<[[TYPE]]>, %[[VAL_1:.*]]: !fir.ref<[[TYPE]]>):
! CHECK:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.type<_QFdelayed_privatization_default_initTt{i:i32}>>) -> !fir.box<!fir.type<_QFdelayed_privatization_default_initTt{i:i32}>>
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.box<!fir.type<_QFdelayed_privatization_default_initTt{i:i32}>>) -> !fir.box<none>
! CHECK:           fir.call @_FortranAInitialize(%[[VAL_6]],{{.*}}
! CHECK:           omp.yield(%[[VAL_1]] : !fir.ref<!fir.type<_QFdelayed_privatization_default_initTt{i:i32}>>)
! CHECK:   }
