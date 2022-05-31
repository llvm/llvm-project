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

! CHECK-LABEL: func @_QPss4{{.*}} {
! CHECK:       omp.parallel {
! CHECK:         omp.wsloop for (%[[ARG:.*]]) : {{.*}} {
! CHECK:           cond_br %{{.*}}, ^bb1, ^bb2
! CHECK:          ^bb1:
! CHECK:           @_FortranAioBeginExternalListOutput
! CHECK:           @_FortranAioOutputInteger32(%{{.*}}, %[[ARG]])
! CHECK:           br ^bb2
! CHECK:         ^bb2:
! CHECK-NEXT:      omp.yield
! CHECK-NEXT:  }
! CHECK:       omp.terminator
! CHECK-NEXT:}
subroutine ss4(n) ! CYCLE in OpenMP wsloop constructs
  !$omp parallel
    do i = 1, 3
      !$omp do
        do j = 1, 3
           if (j .eq. n) cycle
           print*, 'ss4', j
        enddo
      !$omp end do
    enddo
  !$omp end parallel
end

! CHECK-LABEL: func @_QPss5() {
! CHECK:  omp.parallel  {
! CHECK:    omp.wsloop {{.*}} {
! CHECK:      br ^[[BB1:.*]]
! CHECK:    ^[[BB1]]:
! CHECK:      cond_br %{{.*}}, ^[[BB2:.*]], ^[[BB4:.*]]
! CHECK:    ^[[BB2]]:
! CHECK:      cond_br %{{.*}}, ^[[BB4]], ^[[BB3:.*]]
! CHECK:    ^[[BB3]]:
! CHECK:      br ^[[BB1]]
! CHECK:    ^[[BB4]]:
! CHECK:      omp.yield
! CHECK:    }
! CHECK:    omp.terminator
! CHECK:  }
subroutine ss5() ! EXIT inside OpenMP wsloop (inside parallel)
  integer :: x
  !$omp parallel private(x)
    !$omp do
      do j = 1, 3
        x = j * i
        do k = 1, 3
          if (k .eq. n) exit
          x = k
          x = x + k
        enddo
        x = j - 222
      enddo
    !$omp end do
  !$omp end parallel
end

! CHECK-LABEL: func @_QPss6() {
! CHECK:  omp.parallel  {
! CHECK:    br ^[[BB1_OUTER:.*]]
! CHECK:  ^[[BB1_OUTER]]:
! CHECK:    cond_br %{{.*}}, ^[[BB2_OUTER:.*]], ^[[BB3_OUTER:.*]]
! CHECK:  ^[[BB2_OUTER]]:
! CHECK:    omp.wsloop {{.*}} {
! CHECK:      br ^[[BB1:.*]]
! CHECK:    ^[[BB1]]:
! CHECK:      cond_br %{{.*}}, ^[[BB2:.*]], ^[[BB4:.*]]
! CHECK:    ^[[BB2]]:
! CHECK:      cond_br %{{.*}}, ^[[BB4]], ^[[BB3:.*]]
! CHECK:    ^[[BB3]]:
! CHECK:      br ^[[BB1]]
! CHECK:    ^[[BB4]]:
! CHECK:      omp.yield
! CHECK:    }
! CHECK:    br ^[[BB1_OUTER]]
! CHECK:  ^[[BB3_OUTER]]:
! CHECK:    omp.terminator
! CHECK:  }
subroutine ss6() ! EXIT inside OpenMP wsloop in a do loop (inside parallel)
  integer :: x
  !$omp parallel private(x)
    do i = 1, 3
      !$omp do
        do j = 1, 3
          x = j * i
          do k = 1, 3
            if (k .eq. n) exit
            x = k
            x = x + k
          enddo
          x = j - 222
        enddo
      !$omp end do
    enddo
  !$omp end parallel
end

! CHECK-LABEL: func @_QPss7() {
! CHECK: br ^[[BB1_OUTER:.*]]
! CHECK: ^[[BB1_OUTER]]:
! CHECK:   cond_br %{{.*}}, ^[[BB2_OUTER:.*]], ^[[BB3_OUTER:.*]]
! CHECK-NEXT: ^[[BB2_OUTER:.*]]:
! CHECK:   omp.parallel  {
! CHECK:     omp.wsloop {{.*}} {
! CHECK:       br ^[[BB1:.*]]
! CHECK-NEXT:     ^[[BB1]]:
! CHECK:       cond_br %{{.*}}, ^[[BB2:.*]], ^[[BB4:.*]]
! CHECK-NEXT:     ^[[BB2]]:
! CHECK:       cond_br %{{.*}}, ^[[BB4]], ^[[BB3:.*]]
! CHECK-NEXT:     ^[[BB3]]:
! CHECK:       br ^bb1
! CHECK-NEXT:     ^[[BB4]]:
! CHECK:       omp.yield
! CHECK:     }
! CHECK:     omp.terminator
! CHECK:   }
! CHECK:   br ^[[BB1_OUTER]]
! CHECK-NEXT: ^[[BB3_OUTER]]:
! CHECK-NEXT:   return
subroutine ss7() ! EXIT inside OpenMP parallel do (inside do loop)
  integer :: x
    do i = 1, 3
      !$omp parallel do private(x)
        do j = 1, 3
          x = j * i
          do k = 1, 3
            if (k .eq. n) exit
            x = k
            x = x + k
          enddo
        enddo
      !$omp end parallel do
    enddo
end

! CHECK-LABEL: func @_QPss8() {
! CHECK:  omp.parallel  {
! CHECK:    omp.wsloop {{.*}} {
! CHECK:      br ^[[BB1:.*]]
! CHECK:    ^[[BB1]]:
! CHECK:      cond_br %{{.*}}, ^[[BB2:.*]], ^[[BB4:.*]]
! CHECK:    ^[[BB2]]:
! CHECK:      cond_br %{{.*}}, ^[[BB4]], ^[[BB3:.*]]
! CHECK:    ^[[BB3]]:
! CHECK:      br ^[[BB1]]
! CHECK:    ^[[BB4]]:
! CHECK:      omp.yield
! CHECK:    }
! CHECK:    omp.terminator
! CHECK:  }
subroutine ss8() ! EXIT inside OpenMP parallel do
  integer :: x
      !$omp parallel do private(x)
        do j = 1, 3
          x = j * i
          do k = 1, 3
            if (k .eq. n) exit
            x = k
            x = x + k
          enddo
        enddo
      !$omp end parallel do
end

! CHECK-LABEL: func @_QPss9() {
! CHECK:  omp.parallel  {
! CHECK-NEXT:    omp.parallel  {
! CHECK:      br ^[[BB1:.*]]
! CHECK:         ^[[BB1]]:
! CHECK:      cond_br %{{.*}}, ^[[BB2:.*]], ^[[BB4:.*]]
! CHECK-NEXT:    ^[[BB2]]:
! CHECK:      cond_br %{{.*}}, ^[[BB4]], ^[[BB3:.*]]
! CHECK-NEXT:    ^[[BB3]]:
! CHECK:      br ^[[BB1]]
! CHECK-NEXT:    ^[[BB4]]:
! CHECK:      omp.terminator
! CHECK-NEXT:    }
! CHECK:    omp.terminator
! CHECK-NEXT  }
! CHECK: }
subroutine ss9() ! EXIT inside OpenMP parallel (inside parallel)
  integer :: x
  !$omp parallel
  !$omp parallel private(x)
    do k = 1, 3
      if (k .eq. n) exit
      x = k
      x = x + k
    end do
  !$omp end parallel
  !$omp end parallel
end

! CHECK-LABEL: func @_QQmain
program p
  call ss1(2)
  call ss2(2)
  call ss3(2)
  call ss4(2)
  call ss5()
  call ss6()
  call ss7()
  call ss8()
  call ss9()
end
