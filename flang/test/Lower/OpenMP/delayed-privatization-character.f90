! Test delayed privatization for the `CHARACTER` type.

! RUN: split-file %s %t

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %t/dyn_len.f90 2>&1 | FileCheck %s --check-prefix=DYN_LEN
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %t/dyn_len.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=DYN_LEN

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %t/static_len.f90 2>&1 | FileCheck %s --check-prefix=STATIC_LEN
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %t/static_len.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=STATIC_LEN

!--- dyn_len.f90
subroutine delayed_privatization_character(var1, l)
  implicit none
  integer(8):: l
  character(len = l)  :: var1

!$omp parallel firstprivate(var1)
  var1 = "test"
!$omp end parallel
end subroutine

! DYN_LEN-LABEL: omp.private {type = firstprivate}
! DYN_LEN-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.boxchar<1>]] init {

! DYN_LEN-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]], %[[ALLOC_ARG:.*]]: [[TYPE]]):
! DYN_LEN-NEXT:   %[[UNBOX:.*]]:2 = fir.unboxchar %[[PRIV_ARG]]
! DYN_LEN-NEXT:   %[[PRIV_ALLOC:.*]] = fir.allocmem !fir.char<1,?>(%[[UNBOX]]#1 : index)
! DYN_LEN-NEXT:   %[[EMBOXCHAR:.*]] = fir.emboxchar %[[PRIV_ALLOC]], %[[UNBOX]]#1
! DYN_LEN:        omp.yield(%[[EMBOXCHAR]] : !fir.boxchar<1>)

! DYN_LEN-NEXT: } copy {
! DYN_LEN-NEXT: ^bb0(%[[PRIV_ORIG_ARG:.*]]: [[TYPE]], %[[PRIV_PRIV_ARG:.*]]: [[TYPE]]):

! DYN_LEN-NEXT:   hlfir.assign %[[PRIV_ORIG_ARG]] to %[[PRIV_PRIV_ARG]]

! DYN_LEN-NEXT:   omp.yield(%[[PRIV_PRIV_ARG]] : !fir.boxchar<1>)
! DYN_LEN-NEXT: }

!--- static_len.f90
subroutine delayed_privatization_character_static_len(var1)
  implicit none
  character(len = 10)  :: var1

!$omp parallel private(var1)
  var1 = "test"
!$omp end parallel
end subroutine

! STATIC_LEN-LABEL: omp.private {type = private}
! STATIC_LEN-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.char<1,10>]]
