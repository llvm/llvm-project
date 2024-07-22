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
! DYN_LEN-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.boxchar<1>]] alloc {

! DYN_LEN-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):
! DYN_LEN-NEXT:   %[[UNBOX:.*]]:2 = fir.unboxchar %[[PRIV_ARG]]
! DYN_LEN:        %[[PRIV_ALLOC:.*]] = fir.alloca !fir.char<1,?>(%[[UNBOX]]#1 : index)
! DYN_LEN-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]] typeparams %[[UNBOX]]#1
! DYN_LEN-NEXT:   omp.yield(%[[PRIV_DECL]]#0 : !fir.boxchar<1>)

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
! STATIC_LEN-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.ref<!fir.char<1,10>>]] alloc {

! STATIC_LEN-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):
! STATIC_LEN-NEXT:   %[[C10:.*]] = arith.constant 10 : index
! STATIC_LEN-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca !fir.char<1,10>
! STATIC_LEN-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]] typeparams %[[C10]]
