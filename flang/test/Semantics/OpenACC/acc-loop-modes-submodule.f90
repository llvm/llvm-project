! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/parent.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only -J%t %t/child.f90 2>&1 | FileCheck %s --implicit-check-not='warning:'
! RUN: not %flang_fc1 -fopenacc -fsyntax-only -Werror -J%t %t/child.f90 2>&1 | FileCheck %s --implicit-check-not='warning:'
! RUN: %flang_fc1 -fopenacc -fsyntax-only -Werror -w -J%t %t/child.f90 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty
! RUN: %flang_fc1 -fopenacc -fsyntax-only -Werror -Wno-openacc-usage -J%t %t/child.f90 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty
! RUN: not %flang_fc1 -fopenacc -fsyntax-only -Werror -Wno-openacc-usage -Wopenacc-usage -J%t %t/child.f90 2>&1 | FileCheck %s --implicit-check-not='warning:'

! Recovery warnings refer to executable submodule source even when its parent
! scope was imported from a .mod file. Check both warning sites on standalone
! and combined loops, including warning severity and suppression.
! QUIET-NOT: warning:
! QUIET-NOT: error:

!--- parent.f90
module loop_modes_parent
  interface
    module subroutine work(a)
      real :: a(8)
    end subroutine
  end interface
end module

!--- child.f90
submodule(loop_modes_parent) loop_modes_child
contains
  module procedure work
    integer :: i
    !$acc kernels
    !$acc loop seq seq
    do i = 1, 8
      a(i) = i
    end do
    !$acc loop independent independent
    do i = 1, 8
      a(i) = i
    end do
    !$acc loop seq independent
    do i = 1, 8
      a(i) = i
    end do
    !$acc end kernels

    !$acc parallel loop seq seq
    do i = 1, 8
      a(i) = i
    end do
    !$acc parallel loop independent independent
    do i = 1, 8
      a(i) = i
    end do
    !$acc parallel loop seq independent
    do i = 1, 8
      a(i) = i
    end do

    !$acc kernels loop seq seq
    do i = 1, 8
      a(i) = i
    end do
    !$acc kernels loop independent independent
    do i = 1, 8
      a(i) = i
    end do
    !$acc kernels loop seq independent
    do i = 1, 8
      a(i) = i
    end do

    !$acc serial loop seq seq
    do i = 1, 8
      a(i) = i
    end do
    !$acc serial loop independent independent
    do i = 1, 8
      a(i) = i
    end do
    !$acc serial loop seq independent
    do i = 1, 8
      a(i) = i
    end do
  end procedure
end submodule

! CHECK: warning: Repeated SEQ clauses on the LOOP directive are ignored [-Wopenacc-usage]
! CHECK: warning: Repeated INDEPENDENT clauses on the LOOP directive are ignored [-Wopenacc-usage]
! CHECK: warning: SEQ and INDEPENDENT clauses on the LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
! CHECK: warning: Repeated SEQ clauses on the PARALLEL LOOP directive are ignored [-Wopenacc-usage]
! CHECK: warning: Repeated INDEPENDENT clauses on the PARALLEL LOOP directive are ignored [-Wopenacc-usage]
! CHECK: warning: SEQ and INDEPENDENT clauses on the PARALLEL LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
! CHECK: warning: Repeated SEQ clauses on the KERNELS LOOP directive are ignored [-Wopenacc-usage]
! CHECK: warning: Repeated INDEPENDENT clauses on the KERNELS LOOP directive are ignored [-Wopenacc-usage]
! CHECK: warning: SEQ and INDEPENDENT clauses on the KERNELS LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
! CHECK: warning: Repeated SEQ clauses on the SERIAL LOOP directive are ignored [-Wopenacc-usage]
! CHECK: warning: Repeated INDEPENDENT clauses on the SERIAL LOOP directive are ignored [-Wopenacc-usage]
! CHECK: warning: SEQ and INDEPENDENT clauses on the SERIAL LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
