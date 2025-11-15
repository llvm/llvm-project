!RUN: rm -rf %t && mkdir %t && cd %t && \
!RUN:   bbc %s -fopenacc -emit-hlfir -o - \
!RUN:   | fir-opt --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data)" \
!RUN:   | FileCheck %s

! This test exercises whether the ACCImplicitData pass inserts its new
! data operations in appropriate position so that parents are copied in before
! their children.

module types
  type derivc8r4
    complex(8) :: member0
    real(4) :: member1
  end type derivc8r4
end module
program test
  use types
  implicit none
  type (derivc8r4) :: d2
  type (derivc8r4) :: d4
  integer(4) :: i0
  d2%member0 = 123
  !$acc serial copyin(d2%member0) copyout(d4%member0)
  do i0 = 1, 1
    d4%member0 = d2%member0
  end do
  !$acc end serial
end program

!CHECK: acc.copyin {{.*}} {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "d2"}
!CHECK: acc.copyin {{.*}} {name = "d2%member0"}
!CHECK: acc.copyin {{.*}} {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "d4"}
!CHECK: acc.create {{.*}} {dataClause = #acc<data_clause acc_copyout>, name = "d4%member0"}
!CHECK: acc.delete {{.*}} {dataClause = #acc<data_clause acc_copyin>, name = "d2%member0"}
!CHECK: acc.copyout {{.*}} {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "d2"}
!CHECK: acc.copyout {{.*}} {name = "d4%member0"}
!CHECK: acc.copyout {{.*}} {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "d4"}

