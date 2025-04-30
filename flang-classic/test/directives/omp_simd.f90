! RUN: %flang -fopenmp -O2 -S -emit-llvm %s -o - | FileCheck %s
! RUN: %flang -fopenmp -S -emit-llvm %s -o - | FileCheck %s -check-prefix=METADATA

subroutine sum(myarr1,myarr2,ub)
  integer, pointer :: myarr1(:)
  integer, pointer :: myarr2(:)
  integer :: ub

  !$omp simd
  do i=1,ub
    myarr1(i) = myarr1(i)+myarr2(i)
  end do
end subroutine

! CHECK:  {{.*}} add nsw <[[VF:[0-9]+]] x i32>{{.*}}
! METADATA: load {{.*}}, !llvm.access.group ![[TAG1:[0-9]+]]
! METADATA: store {{.*}}, !llvm.access.group ![[TAG1]]
! METADATA: ![[TAG1]] = distinct !{}
! METADATA: ![[TAG4:[0-9]+]] = distinct !{![[TAG4]], ![[TAG2:[0-9]+]], ![[TAG3:[0-9]+]], {{.*}}, {{.*}}}
! METADATA: ![[TAG2]] = !{!"llvm.loop.vectorize.enable", i1 true}
! METADATA: ![[TAG3]] = !{!"llvm.loop.parallel_accesses", ![[TAG1:[0-9]+]]}
