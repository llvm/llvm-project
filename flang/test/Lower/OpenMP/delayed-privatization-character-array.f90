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
! STATIC_LEN-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.ref<!fir.array<5x!fir.char<1,10>>>]] alloc {

! STATIC_LEN-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):
! STATIC_LEN-DAG:    %[[C5:.*]] = arith.constant 5 : index
! STATIC_LEN-DAG:    %[[C10:.*]] = arith.constant 10 : index
! STATIC_LEN-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca !fir.array<5x!fir.char<1,10>>
! STATIC_LEN-NEXT:   %[[ARRAY_SHAPE:.*]] = fir.shape %[[C5]]
! STATIC_LEN-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]](%[[ARRAY_SHAPE]]) typeparams %[[C10]]
! STATIC_LEN-NEXT:   omp.yield(%[[PRIV_DECL]]#0

! STATIC_LEN-NEXT: } copy {
! STATIC_LEN-NEXT: ^bb0(%[[PRIV_ORIG_ARG:.*]]: [[TYPE]], %[[PRIV_PRIV_ARG:.*]]: [[TYPE]]):
! STATIC_LEN-NEXT:   hlfir.assign %[[PRIV_ORIG_ARG]] to %[[PRIV_PRIV_ARG]]

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
! DYN_LEN-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.box<!fir.array<\?x!fir.char<1,\?>>>]] alloc {

! DYN_LEN-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):

! DYN_LEN:        %[[C0:.*]] = arith.constant 0 : index
! DYN_LEN-NEXT:   %[[BOX_DIM:.*]]:3 = fir.box_dims %[[PRIV_ARG]], %[[C0]]
! DYN_LEN:        %[[CHAR_LEN:.*]] = fir.box_elesize %[[PRIV_ARG]]
! DYN_LEN-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca !fir.array<?x!fir.char<1,?>>(%[[CHAR_LEN]] : index)
! DYN_LEN-NEXT:   %[[ARRAY_SHAPE:.*]] = fir.shape
! DYN_LEN-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]](%[[ARRAY_SHAPE]]) typeparams %[[CHAR_LEN]]

! DYN_LEN-NEXT:   omp.yield(%[[PRIV_DECL]]#0
