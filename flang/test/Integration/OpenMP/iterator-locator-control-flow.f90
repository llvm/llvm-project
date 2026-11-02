!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! Iterator locators are lowered inside the omp.iterator region. Lowering and
! later passes can add control flow there, which only shows up after
! control-flow conversion, so these cases are checked through LLVM IR.

! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=52 -o - %s \
! RUN:   | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -O1 -fopenmp -fopenmp-version=52 -o - %s \
! RUN:   | FileCheck %s --check-prefix=O1

module m
contains
  function fa(i) result(r)
    integer, intent(in) :: i
    integer, allocatable :: r
    allocate(r)
    r = i + 1
  end function
end module

! LEN_TRIM in the section bound lowers to a loop.
subroutine depend_section_len_trim(a, s, n)
  integer :: a(10), n
  character(*) :: s
  !$omp task depend(iterator(i=1:n), in: a(i:len_trim(s)))
  !$omp end task
end

! CHECK-LABEL: define void @depend_section_len_trim_(
! CHECK-SAME: ptr {{[^%]*}}%[[A:[0-9]+]],
! CHECK: omp.iterator.region:
! CHECK: phi i64
! CHECK: getelementptr [10 x i32], ptr %[[A]]
! CHECK: %[[P:.*]] = load ptr, ptr
! CHECK: omp.iterator.region.cont:
! CHECK: %[[PI:.*]] = ptrtoint ptr %[[P]] to i64
! CHECK: store i64 %[[PI]], ptr
! CHECK: br label %omp_dep_iterator.inc
! CHECK: call i32 @__kmpc_omp_task_with_deps(

! The affinity length spans the section up to LEN_TRIM(s).
subroutine affinity_section_len_trim(a, s, n)
  integer :: a(10), n
  character(*) :: s
  !$omp task affinity(iterator(i=1:n): a(i:len_trim(s)))
  !$omp end task
end

! CHECK-LABEL: define void @affinity_section_len_trim_(
! CHECK-SAME: ptr {{[^%]*}}%[[A:[0-9]+]],
! CHECK: omp.iterator.region:
! CHECK: phi i64
! CHECK: getelementptr [10 x i32], ptr %[[A]]
! CHECK: %[[P:.*]] = load ptr, ptr
! CHECK: %[[LEN:.*]] = mul i64 %{{.*}}, 4
! CHECK: omp.iterator.region.cont:
! CHECK: %[[PI:.*]] = ptrtoint ptr %[[P]] to i64
! CHECK: store i64 %[[PI]], ptr
! CHECK: store i64 %[[LEN]], ptr
! CHECK: br label %omp_iterator.inc
! CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(

! The allocatable function result is freed conditionally.
subroutine depend_alloc_result(a, n)
  use m
  integer :: a(10), n
  !$omp task depend(iterator(i=1:n), in: a(i:fa(i)))
  !$omp end task
end

! CHECK-LABEL: define void @depend_alloc_result_(
! CHECK-SAME: ptr {{[^%]*}}%[[A:[0-9]+]],
! CHECK: omp.iterator.region:
! CHECK: call void @_QMmPfa(
! CHECK: getelementptr [10 x i32], ptr %[[A]]
! CHECK: %[[P:.*]] = load ptr, ptr
! CHECK: omp.iterator.region{{[0-9]+}}:
! CHECK: call void @free(
! CHECK: omp.iterator.region.cont:
! CHECK: %[[PI:.*]] = ptrtoint ptr %[[P]] to i64
! CHECK: store i64 %[[PI]], ptr
! CHECK: call i32 @__kmpc_omp_task_with_deps(

! At -O1, SUM is inlined into the region of a folded locator.
subroutine depend_folded_sum(a, v, n)
  integer :: a(10), v(3), n
  !$omp task depend(iterator(i=1:n), in: a(sum(v)+0*i))
  !$omp end task
end

! O1-LABEL: define void @depend_folded_sum_(
! O1-SAME: ptr {{[^%]*}}%[[A:[0-9]+]], ptr {{[^%]*}}%[[V:[0-9]+]],
! O1: omp.iterator.region{{[0-9]*}}:
! O1: getelementptr {{.*}}ptr %[[V]]
! O1: omp.iterator.region{{[0-9]*}}:
! O1: getelementptr {{.*}}ptr %[[A]]
! O1: call i32 @__kmpc_omp_task_with_deps(
