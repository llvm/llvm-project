! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa %s -o - | FileCheck %s

! CHECK-DAG: c";arr;
subroutine fp_array(res)
  integer :: res, arr(4)
  arr = (/1, 2, 3, 4/)
  !$omp target map(tofrom: res) firstprivate(arr)
    res = arr(1)
  !$omp end target
end subroutine

! CHECK-DAG: c";cstr;
subroutine fp_char(res)
  integer :: res
  character(len=4) :: cstr
  cstr = "abcd"
  !$omp target map(tofrom: res) firstprivate(cstr)
    res = ichar(cstr(1:1))
  !$omp end target
end subroutine

! CHECK-DAG: c";pa;
subroutine priv_alloc(res)
  integer :: res
  integer, allocatable :: pa(:)
  allocate(pa(4))
  !$omp target map(tofrom: res) private(pa)
    res = 9
  !$omp end target
  deallocate(pa)
end subroutine
