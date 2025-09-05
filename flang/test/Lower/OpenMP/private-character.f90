!RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: func @_QPtest_dynlen_char_ptr
!CHECK:         omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[A:.*]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) {
!CHECK:           %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_dynlen_char_ptrEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>)
!CHECK:           %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
!CHECK:           %[[LEN:.*]] = fir.box_elesize %[[A_VAL]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
!CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_DECL]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
!CHECK:           %[[LEN_I64:.*]] = fir.convert %[[LEN]] : (index) -> i64
!CHECK:           fir.call @_FortranAPointerNullifyCharacter(%[[A_BOX_NONE]], %[[LEN_I64]], {{.*}})
subroutine test_dynlen_char_ptr(i)
  character(i), pointer :: a

  !$omp parallel private(a)
    allocate(a)
    a = "abc"
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPtest_dynlen_char_ptr_array
!CHECK:         omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[A:.*]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) {
!CHECK:           %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_dynlen_char_ptr_arrayEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>)
!CHECK:           %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0
!CHECK:           %[[LEN:.*]] = fir.box_elesize %[[A_VAL]]
!CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_DECL]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
!CHECK:           %[[LEN_I64:.*]] = fir.convert %[[LEN]] : (index) -> i64
!CHECK:           fir.call @_FortranAPointerNullifyCharacter(%[[A_BOX_NONE]], %[[LEN_I64]], {{.*}})
subroutine test_dynlen_char_ptr_array(i)
  character(i), pointer :: a(:)

  !$omp parallel private(a)
    allocate(a(i))
    a = "abc"
  !$omp end parallel
end subroutine
