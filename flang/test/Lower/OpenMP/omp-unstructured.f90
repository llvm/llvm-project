! Test unstructured code adjacent to and inside OpenMP constructs.

! RUN: bbc %s -fopenmp -o "-" | FileCheck %s

! CHECK-LABEL: func @_QPss1{{.*}} {
! CHECK:   br ^bb1
! CHECK: ^bb1:  // 2 preds: ^bb0, ^bb3
! CHECK:   cond_br %{{[0-9]*}}, ^bb2, ^bb4
! CHECK: ^bb2:  // pred: ^bb1
! CHECK:   cond_br %{{[0-9]*}}, ^bb4, ^bb3
! CHECK: ^bb3:  // pred: ^bb2
! CHECK:   @_FortranAioBeginExternalListOutput
! CHECK:   br ^bb1
! CHECK: ^bb4:  // 2 preds: ^bb1, ^bb2
! CHECK:   omp.master  {
! CHECK:     @_FortranAioBeginExternalListOutput
! CHECK:     omp.terminator
! CHECK:   }
! CHECK:   @_FortranAioBeginExternalListOutput
! CHECK: }
subroutine ss1(n) ! unstructured code followed by a structured OpenMP construct
  do i = 1, 3
    if (i .eq. n) exit
    print*, 'ss1-A', i
  enddo
  !$omp master
    print*, 'ss1-B', i
  !$omp end master
  print*
end

! CHECK-LABEL: func @_QPss2{{.*}} {
! CHECK:   omp.master  {
! CHECK:     @_FortranAioBeginExternalListOutput
! CHECK:     br ^bb1
! CHECK:   ^bb1:  // 2 preds: ^bb0, ^bb3
! CHECK:     cond_br %{{[0-9]*}}, ^bb2, ^bb4
! CHECK:   ^bb2:  // pred: ^bb1
! CHECK:     cond_br %{{[0-9]*}}, ^bb4, ^bb3
! CHECK:   ^bb3:  // pred: ^bb2
! CHECK:     @_FortranAioBeginExternalListOutput
! CHECK:     br ^bb1
! CHECK:   ^bb4:  // 2 preds: ^bb1, ^bb2
! CHECK:     omp.terminator
! CHECK:   }
! CHECK:   @_FortranAioBeginExternalListOutput
! CHECK:   @_FortranAioBeginExternalListOutput
! CHECK: }
subroutine ss2(n) ! unstructured OpenMP construct; loop exit inside construct
  !$omp master
    print*, 'ss2-A', n
    do i = 1, 3
      if (i .eq. n) exit
      print*, 'ss2-B', i
    enddo
  !$omp end master
  print*, 'ss2-C', i
  print*
end

! CHECK-LABEL: func @_QPss3{{.*}} {
! CHECK:   omp.parallel  {
! CHECK:     br ^bb1
! CHECK:   ^bb1:  // 2 preds: ^bb0, ^bb2
! CHECK:     cond_br %{{[0-9]*}}, ^bb2, ^bb3
! CHECK:   ^bb2:  // pred: ^bb1
! CHECK:     omp.wsloop {{.*}} {
! CHECK:     @_FortranAioBeginExternalListOutput
! CHECK:       omp.yield
! CHECK:     }
! CHECK:     omp.wsloop {{.*}} {
! CHECK:       br ^bb1
! CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
! CHECK:       cond_br %{{[0-9]*}}, ^bb2, ^bb4
! CHECK:     ^bb2:  // pred: ^bb1
! CHECK:       cond_br %{{[0-9]*}}, ^bb4, ^bb3
! CHECK:     ^bb3:  // pred: ^bb2
! CHECK:       @_FortranAioBeginExternalListOutput
! CHECK:       br ^bb1
! CHECK:     ^bb4:  // 2 preds: ^bb1, ^bb2
! CHECK:       omp.yield
! CHECK:     }
! CHECK:     br ^bb1
! CHECK:   ^bb3:  // pred: ^bb1
! CHECK:     omp.terminator
! CHECK:   }
! CHECK: }
subroutine ss3(n) ! nested unstructured OpenMP constructs
  !$omp parallel
    do i = 1, 3
      !$omp do
        do k = 1, 3
          print*, 'ss3-A', k
        enddo
      !$omp end do
      !$omp do
        do j = 1, 3
          do k = 1, 3
            if (k .eq. n) exit
            print*, 'ss3-B', k
          enddo
        enddo
      !$omp end do
    enddo
  !$omp end parallel
end

! CHECK-LABEL: func @_QQmain
program p
  call ss1(2)
  call ss2(2)
  call ss3(2)
end
