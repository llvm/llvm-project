!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-device %s -o - | FileCheck %s

!CHECK: func.func @{{.*}}b()
!CHECK-NOT: omp.declare_target

!CHECK: func.func @{{.*}}c()
!CHECK-SAME: omp.declare_target

!CHECK: func.func private @{{.*}}a()
!CHECK-SAME: omp.declare_target

module iface
interface
  subroutine a()
  !$omp declare target
  end subroutine
end interface
contains
  subroutine b()
    call a()
  end subroutine
  subroutine c()
    !$omp declare target
    call a()
  end subroutine
end module
