! Verify that a module re-exporting DEC STRUCTUREs through a plain USE writes a
! module file that can be read back by a separate compilation, and that
! renames and accessibility that hide other names are preserved.

! RUN: rm -rf %t && split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m1.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m2.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m3.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/other.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/facade.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/facade2.f90
! RUN: FileCheck --check-prefix=M2 --input-file=%t/mf91_m2.mod %s
! RUN: FileCheck --check-prefix=M3 --input-file=%t/mf91_m3.mod %s
! RUN: FileCheck --check-prefix=FACADE --input-file=%t/mf91_facade.mod %s
! RUN: FileCheck --check-prefix=FACADE2 --input-file=%t/mf91_facade2.mod %s
! RUN: %flang_fc1 -fsyntax-only -J%t %t/p.f90
! RUN: not %flang_fc1 -fsyntax-only -J%t %t/bad.f90 2>&1 | FileCheck --check-prefix=BAD %s
! RUN: %flang_fc1 -fsyntax-only -J%t %t/host.f90
! RUN: not %flang_fc1 -fsyntax-only -J%t %t/host2.f90 2>&1 | FileCheck --check-prefix=HOST2 %s

! M2-DAG: {{^}}use mf91_m1{{$}}
! M2-DAG: {{^}}use mf91_m1,only:xx=>x{{$}}
! M2-NOT: /s/
! M3-NOT: {{^}}use mf91_m1{{$}}
! FACADE-DAG: {{^}}use mf91_m1{{$}}
! FACADE-DAG: {{^}}use mf91_m1,only:z=>x{{$}}
! FACADE-DAG: {{^}}use mf91_other,only:z=>y{{$}}
! FACADE2-DAG: {{^}}use mf91_m1,only:z=>x{{$}}
! FACADE2-DAG: {{^}}use mf91_other,only:z=>y{{$}}
! FACADE2-DAG: private::z
! BAD: error: No explicit type declared for 'x'
! HOST2: error: Reference to 'z' is ambiguous

!--- m1.f90
module mf91_m1
  structure /s/
    integer :: i
    structure inner
      real :: r
    end structure
  end structure
  integer, parameter :: x = 1, y = 2
end module

!--- m2.f90
module mf91_m2
  use mf91_m1, xx => x
end module

!--- m3.f90
module mf91_m3
  use mf91_m1
  private
  public :: y
end module

!--- other.f90
module mf91_other
  integer, parameter :: y = 3
end module

!--- facade.f90
module mf91_facade
  use mf91_m1, z => x
  use mf91_other, z => y
end module

!--- facade2.f90
module mf91_facade2
  use mf91_m1, z => x
  use mf91_other, z => y
  private :: z
end module

!--- p.f90
program p
  use mf91_m2
  use mf91_m3
  implicit none
  print *, xx, y
end program

!--- bad.f90
program bad
  use mf91_m2
  implicit none
  print *, x
end program

!--- host.f90
program host
  implicit none
  integer, parameter :: x = 99
  call show
contains
  subroutine show
    use mf91_facade
    ! x must still be the host's x (99): the facade renamed the module's x.
    integer, parameter :: chk(x - 98) = [5]
  end subroutine
end program

!--- host2.f90
program host2
  use mf91_facade
  implicit none
  print *, z
end program
