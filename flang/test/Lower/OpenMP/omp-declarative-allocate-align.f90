! This test checks lowering of OpenMP allocate Directive with align and allocator
! clauses to HLFIR. Verifies code generation for:
!   - align(16) only (null allocator)
!   - allocator(1) only (no align)
!   - align(64) allocator(6) (both clauses, array variable)
!   - align(32) allocator(3) (both clauses, multiple variables)
! Each omp.allocate_dir must be paired with a matching omp.allocate_free

! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 %s -o - 2>&1 | FileCheck %s

program main
  integer :: x, y
  integer :: z(10)
  character c
  real :: r
  complex :: cmplx
  !$omp allocate(x) align(16)
  !$omp allocate(y) allocator(1)
  !$omp allocate(z) align(64) allocator(6)
  !$omp allocate(c, r, cmplx) align(32) allocator(3)
  x = 1
  y = 2
  z = x + y
  print *, "z : ", z
end program

! CHECK: %[[C1_IDX:.*]] = arith.constant 1 : index
! CHECK: %[[C_ALLOC:.*]] = fir.alloca !fir.char<1> {bindc_name = "c", uniq_name = "_QFEc"}
! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C_ALLOC]] typeparams %[[C1_IDX]] {uniq_name = "_QFEc"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
! CHECK: %[[CMPLX_ALLOC:.*]] = fir.alloca complex<f32> {bindc_name = "cmplx", uniq_name = "_QFEcmplx"}
! CHECK: %[[CMPLX_DECL:.*]]:2 = hlfir.declare %[[CMPLX_ALLOC]] {uniq_name = "_QFEcmplx"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
! CHECK: %[[R_ALLOC:.*]] = fir.alloca f32 {bindc_name = "r", uniq_name = "_QFEr"}
! CHECK: %[[R_DECL:.*]]:2 = hlfir.declare %[[R_ALLOC]] {uniq_name = "_QFEr"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK: %[[X_ALLOC:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_ALLOC]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[Y_ALLOC:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_ALLOC]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[Z_REF:.*]] = fir.address_of(@_QFEz) : !fir.ref<!fir.array<10xi32>>
! CHECK: %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z_REF]]({{.*}}) {uniq_name = "_QFEz"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK: omp.allocate_dir(%[[X_DECL]]#0 : !fir.ref<i32>) align(16)
! CHECK: %[[ALLOC1:.*]] = arith.constant 1 : i32
! CHECK: omp.allocate_dir(%[[Y_DECL]]#0 : !fir.ref<i32>) allocator(%[[ALLOC1]] : i32)
! CHECK: %[[ALLOC6:.*]] = arith.constant 6 : i32
! CHECK: omp.allocate_dir(%[[Z_DECL]]#0 : !fir.ref<!fir.array<10xi32>>) align(64) allocator(%[[ALLOC6]] : i32)
! CHECK: %[[ALLOC3:.*]] = arith.constant 3 : i32
! CHECK: omp.allocate_dir(%[[C_DECL]]#0, %[[R_DECL]]#0, %[[CMPLX_DECL]]#0 : !fir.ref<!fir.char<1>>, !fir.ref<f32>, !fir.ref<complex<f32>>) align(32) allocator(%[[ALLOC3]] : i32)
! CHECK: omp.allocate_free(%[[C_DECL]]#0, %[[R_DECL]]#0, %[[CMPLX_DECL]]#0 : !fir.ref<!fir.char<1>>, !fir.ref<f32>, !fir.ref<complex<f32>>) allocator(%[[ALLOC3]] : i32)
! CHECK: omp.allocate_free(%[[Z_DECL]]#0 : !fir.ref<!fir.array<10xi32>>) allocator(%[[ALLOC6]] : i32)
! CHECK: omp.allocate_free(%[[Y_DECL]]#0 : !fir.ref<i32>) allocator(%[[ALLOC1]] : i32)
! CHECK: omp.allocate_free(%[[X_DECL]]#0 : !fir.ref<i32>)
! CHECK: return
