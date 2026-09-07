! RUN: %flang_fc1 -fopenmp-default-allocate=target -mmlir -use-alloc-runtime -emit-hlfir %s -o - | FileCheck %s

subroutine allocate_deallocate()
  real, allocatable :: x
! CHECK: fir.call @_FortranAOpenMPAllocatableSetAllocIdx({{.*}}, %c1{{[^)]*}}) {{.*}} : (!fir.ref<!fir.box<!fir.heap<f32>>>, i32) -> ()
! CHECK: fir.call @_FortranAAllocatableAllocate
  allocate(x)

! CHECK: fir.call @_FortranAAllocatableDeallocate
  deallocate(x)
end subroutine

subroutine test_allocatable_scalar(a)
  real, save, allocatable :: x1, x2
  real :: a

! CHECK: fir.call @_FortranAOpenMPAllocatableSetAllocIdx({{.*}}, %c1{{[^)]*}}) {{.*}} : (!fir.ref<!fir.box<!fir.heap<f32>>>, i32) -> ()
! CHECK: fir.call @_FortranAAllocatableAllocateSource
  allocate(x1, x2, source = a)
end
