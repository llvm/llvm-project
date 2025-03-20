! Test delayed privatization for the `CHARACTER` array type.

! RUN: split-file %s %t

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %t/static_len.f90 2>&1 | FileCheck %s --check-prefix=STATIC_LEN
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %t/static_len.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=STATIC_LEN

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %t/dyn_len.f90 2>&1 | FileCheck %s --check-prefix=DYN_LEN
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %t/dyn_len.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=DYN_LEN

!--- static_len.f90
subroutine delayed_privatization_character_array_static_len(var1)
  implicit none
  character(len = 10)  :: var1(5)

!$omp parallel firstprivate(var1)
  var1(1) = "test"
!$omp end parallel
end subroutine

! STATIC_LEN-LABEL: omp.private {type = firstprivate}
! STATIC_LEN-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.box<!fir.array<5x!fir.char<1,10>>>]] init {

! STATIC_LEN-NEXT: ^bb0(%[[MOLD_REF:.*]]: !fir.ref<[[TYPE]]>, %[[ALLOC:.*]]: !fir.ref<[[TYPE]]>):
!                    [init region]
! STATIC_LEN:      } copy {
! STATIC_LEN-NEXT: ^bb0(%[[PRIV_ORIG_ARG:.*]]: !fir.ref<[[TYPE]]>, %[[PRIV_PRIV_ARG:.*]]: !fir.ref<[[TYPE]]>):
! STATIC_LEN-NEXT:   %[[ORIG:.*]] = fir.load %[[PRIV_ORIG_ARG]] : !fir.ref<[[TYPE]]>
! STATIC_LEN-NEXT:   hlfir.assign %[[ORIG]] to %[[PRIV_PRIV_ARG]]

! STATIC_LEN-NEXT:   omp.yield(%[[PRIV_PRIV_ARG]]
! STATIC_LEN-NEXT: }

!--- dyn_len.f90
subroutine delayed_privatization_character_array_dynamic_len(var1, char_len, array_len)
  implicit none
  integer(8):: char_len
  integer(8):: array_len
  character(len = char_len)  :: var1(array_len)

!$omp parallel private(var1)
  var1(1) = "test"
!$omp end parallel
end subroutine

! DYN_LEN-LABEL: omp.private {type = private}
! DYN_LEN-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.box<!fir.array<\?x!fir.char<1,\?>>>]] init {
! DYN_LEN-NEXT: ^bb0(%[[MOLD_ARG:.*]]: !fir.ref<!fir.box<!fir.array<?x!fir.char<1,?>>>>, %[[ALLOC_ARG:.*]]: !fir.ref<!fir.box<!fir.array<?x!fir.char<1,?>>>>)
