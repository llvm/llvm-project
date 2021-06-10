! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Test alternat return argument passing for internal and external subprograms
! Both of the following are OK
  call extSubprogram (*100)
  call intSubprogram (*100)
  call extSubprogram (*101)
  call intSubprogram (*101)
100 PRINT *,'First alternate return'
!ERROR: Label '101' is not a branch target
!ERROR: Label '101' is not a branch target
101 FORMAT("abc")
contains
  subroutine intSubprogram(*)
    return(1)
  end subroutine
end
