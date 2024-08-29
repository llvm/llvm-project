! RUN: bbc -pft-test -o %t %s | FileCheck %s

module mm
  !dir$ some directive 1
  type t
    logical :: tag
  contains
    final :: fin
  end type
  !dir$ some directive 2

contains
  !dir$ some directive 3
  subroutine fin(x)
    type(t), intent(inout) :: x
    x%tag =.true.
    !dir$ some directive 4
    call s1
    call s2
    print*, 'fin', x

  contains
    !dir$ some directive 5
    subroutine s1
      print*, 's1'
      !dir$ some directive 6
    end subroutine s1

    !dir$ some directive 7
    subroutine s2
      !dir$ some directive 8
      if (.true.) then
        !dir$ some directive 9
        print*, 's2'
        !dir$ some directive 10
      endif
      !dir$ some directive 11
    end subroutine s2
    !dir$ some directive 12
  end subroutine fin
  !dir$ some directive 13
end module mm
!dir$ some directive 14

end

! CHECK:  Module mm: module mm
! CHECK:    CompilerDirective: !some directive 1
! CHECK:    CompilerDirective: !some directive 2

! CHECK:  Contains
! CHECK:  CompilerDirective: !some directive 3

! CHECK:  Subroutine fin: subroutine fin(x)
! CHECK:    AssignmentStmt: x%tag =.true.
! CHECK:    CompilerDirective: !some directive 4
! CHECK:    CallStmt: call s1
! CHECK:    CallStmt: call s2
! CHECK:    PrintStmt: print*, 'fin', x
! CHECK:    EndSubroutineStmt: end subroutine fin

! CHECK:  Contains
! CHECK:  CompilerDirective: !some directive 5

! CHECK:  Subroutine s1: subroutine s1
! CHECK:    PrintStmt: print*, 's1'
! CHECK:    CompilerDirective: !some directive 6
! CHECK:    EndSubroutineStmt: end subroutine s1
! CHECK:  End Subroutine s1

! CHECK:  CompilerDirective: !some directive 7

! CHECK:  Subroutine s2: subroutine s2
! CHECK:    CompilerDirective: !some directive 8
! CHECK:    <<IfConstruct>> -> 7
! CHECK:      IfThenStmt -> 7: if(.true.) then
! CHECK:      ^CompilerDirective: !some directive 9
! CHECK:      PrintStmt: print*, 's2'
! CHECK:      CompilerDirective: !some directive 10
! CHECK:      EndIfStmt: endif
! CHECK:    <<End IfConstruct>>
! CHECK:    CompilerDirective: !some directive 11
! CHECK:    EndSubroutineStmt: end subroutine s2
! CHECK:  End Subroutine s2

! CHECK:  CompilerDirective: !some directive 12

! CHECK:  End Contains
! CHECK:  End Subroutine fin

! CHECK:  CompilerDirective: !some directive 13

! CHECK:  End Contains
! CHECK:  End Module mm

! CHECK:  CompilerDirective: !some directive 14

! CHECK:  Program <anonymous>
! CHECK:    EndProgramStmt: end
! CHECK:  End Program <anonymous>
