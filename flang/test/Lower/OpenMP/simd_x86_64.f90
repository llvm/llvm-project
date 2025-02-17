! Tests for 2.9.3.1 Simd and target dependent defult alignment for x86
! REQUIRES: x86-registered-target
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-hlfir -fopenmp -target-cpu x86-64  %s -o - | FileCheck --check-prefixes=DEFAULT %s
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-hlfir -fopenmp -target-cpu x86-64 -target-feature +avx %s -o - | FileCheck --check-prefixes=AVX %s
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-hlfir -fopenmp -target-cpu x86-64  -target-feature +avx512f  %s -o - | FileCheck --check-prefixes=AVX512F %s
!DEFAULT: func.func @_QPsimdloop_aligned_cptr(%[[ARG_A:.*]]: !fir.ref
!DEFAULT-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr
!DEFAULT-SAME: {__address:i64}>> {fir.bindc_name = "a"}) {
!DEFAULT:  %[[A_DECL:.*]]:2 = hlfir.declare %[[ARG_A]] dummy_scope %0
!DEFAULT-SAME:  {uniq_name = "_QFsimdloop_aligned_cptrEa"} :
!DEFAULT-SAME:  (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.dscope) ->
!DEFAULT-SAME:  (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>,
!DEFAULT-SAME:  !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
!AVX: func.func @_QPsimdloop_aligned_cptr(%[[ARG_A:.*]]: !fir.ref
!AVX-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr
!AVX-SAME: {__address:i64}>> {fir.bindc_name = "a"}) {
!AVX:  %[[A_DECL:.*]]:2 = hlfir.declare %[[ARG_A]] dummy_scope %0
!AVX-SAME:  {uniq_name = "_QFsimdloop_aligned_cptrEa"} :
!AVX-SAME:  (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.dscope) ->
!AVX-SAME:  (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>,
!AVX-SAME:  !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
!AVX512F: func.func @_QPsimdloop_aligned_cptr(%[[ARG_A:.*]]: !fir.ref
!AVX512F-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr
!AVX512F-SAME: {__address:i64}>> {fir.bindc_name = "a"}) {
!AVX512F:  %[[A_DECL:.*]]:2 = hlfir.declare %[[ARG_A]] dummy_scope %0
!AVX512F-SAME:  {uniq_name = "_QFsimdloop_aligned_cptrEa"} :
!AVX512F-SAME:  (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.dscope) ->
!AVX512F-SAME:  (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>,
!AVX512F-SAME:  !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
subroutine simdloop_aligned_cptr(A)
    use iso_c_binding
    integer :: i
    type (c_ptr) :: A
    !DEFAULT: omp.simd aligned(%[[A_DECL]]#1 : !fir.ref
    !DEFAULT-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
    !DEFAULT-SAME: -> 128 : i64)
    !AVX: omp.simd aligned(%[[A_DECL]]#1 : !fir.ref
    !AVX-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
    !AVX-SAME: -> 256 : i64)
    !AVX512F: omp.simd aligned(%[[A_DECL]]#1 : !fir.ref
    !AVX512F-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
    !AVX512F-SAME: -> 512 : i64)
    !$OMP SIMD ALIGNED(A)
    do i = 1, 10
        call c_test_call(A)
    end do
    !$OMP END SIMD
end subroutine
