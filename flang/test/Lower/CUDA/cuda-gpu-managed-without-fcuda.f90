! RUN: bbc -emit-hlfir -gpu=managed %s -o - | FileCheck %s
! RUN: bbc -emit-hlfir -gpu=mem:managed %s -o - | FileCheck %s

! Test that -gpu=managed (a.k.a. -gpu=mem:managed) does NOT implicitly tag
! plain Fortran allocatables as CUDA managed when CUDA Fortran is not
! enabled. The implicit-managed tagging is a CUDA Fortran convenience and
! should only fire when -fcuda is also in effect (or the source is .cuf).
! Otherwise, a non-CUDA-Fortran translation unit (e.g. pure OpenACC code
! compiled with -gpu=mem:managed by the driver) would route every
! allocatable through the CUDA Fortran managed descriptor pipeline and
! crash at runtime in cudaGetSymbolAddress.

! -----------------------------------------------------------------------------
! Test 1: Plain allocatable stays plain - no cuf.* ops, no managed tagging
! -----------------------------------------------------------------------------
subroutine test_no_implicit_managed()
  real, allocatable :: a(:)
  allocate(a(100))
  a = 1.0
  deallocate(a)
end subroutine

! CHECK-LABEL: func.func @_QPtest_no_implicit_managed()
! CHECK-NOT:   cuf.alloc
! CHECK-NOT:   data_attr = #cuf.cuda<managed>
! CHECK-NOT:   allocator_idx = 3
! CHECK-NOT:   cuf.allocate
! CHECK-NOT:   cuf.deallocate
! CHECK:       fir.call @_FortranAAllocatableAllocate
! CHECK:       fir.call @_FortranAAllocatableDeallocate

! -----------------------------------------------------------------------------
! Test 2: Multi-dimensional allocatable also stays plain
! -----------------------------------------------------------------------------
subroutine test_no_implicit_managed_multidim()
  real, allocatable :: arr(:,:,:)
  allocate(arr(10,20,30))
  arr = 0.0
  deallocate(arr)
end subroutine

! CHECK-LABEL: func.func @_QPtest_no_implicit_managed_multidim()
! CHECK-NOT:   data_attr = #cuf.cuda<managed>
! CHECK-NOT:   allocator_idx = 3

! -----------------------------------------------------------------------------
! Test 3: Module-level allocatable global also stays plain
! -----------------------------------------------------------------------------
module mod_no_managed
  real, allocatable :: g(:)
end module

subroutine test_no_implicit_managed_module()
  use mod_no_managed
  allocate(g(50))
  deallocate(g)
end subroutine

! CHECK:     fir.global @_QMmod_no_managedEg : !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK-NOT: data_attr = #cuf.cuda<managed>
! CHECK-NOT: allocator_idx = 3
