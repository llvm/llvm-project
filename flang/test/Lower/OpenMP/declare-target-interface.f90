!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-device %s -o - | FileCheck %s

!CHECK: module attributes
module iface
interface
  subroutine a()
  !$omp declare target
  end subroutine
end interface
end module
