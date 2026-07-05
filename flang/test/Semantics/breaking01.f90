! RUN: %flang_fc1 -fsyntax-only -pedantic %s  2>&1 | FileCheck %s --allow-empty
! Verify portability warning on usage that trips over a F202X breaking change
program main
  character(:), allocatable :: str
  real, allocatable :: x
  allocate(character(10)::str)
!CHECK: portability: The deferred length allocatable character scalar variable 'str' may be reallocated to a different length under the new Fortran 202X standard semantics for Internal file
  write(str, 1) 3.14159
1 format(F6.4)
  print 2, str
2 format('>',a,'<')
!CHECK: portability: The deferred length allocatable character scalar variable 'str' may be reallocated to a different length under the new Fortran 202X standard semantics for IOMSG=
  open(1,file="/dev/nonexistent",status="old",iomsg=str)
!CHECK: portability: The deferred length allocatable character scalar variable 'str' may be reallocated to a different length under the new Fortran 202X standard semantics for ENCODING
  inquire(6,encoding=str)
!CHECK: portability: The deferred length allocatable character scalar variable 'str' may be reallocated to a different length under the new Fortran 202X standard semantics for ERRMSG=
  allocate(x,errmsg=str)
!CHECK: portability: The deferred length allocatable character scalar variable 'str' may be reallocated to a different length under the new Fortran 202X standard semantics for ERRMSG=
  deallocate(x,errmsg=str)
!CHECK: portability: The deferred length allocatable character scalar variable 'str' may be reallocated to a different length under the new Fortran 202X standard semantics for dummy argument 'cmdmsg='
  call execute_command_line("true", cmdmsg=str)
end
