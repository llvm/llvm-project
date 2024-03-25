! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in change team statements.
! Only those semantics which differ from those of FORM TEAM statements are checked.

subroutine test
  use, intrinsic :: iso_fortran_env, only: team_type
  type(team_type) :: team
  integer, codimension[*] :: selector
  integer, codimension[2,*] :: selector2d

  ! Valid invocations which should produce no errors.
  block
  change team (team)
  end team
  construct1: change team (team)
  end team construct1
  change team (team, ca[*] => selector)
  end team
  change team (team, ca[2,*] => selector)
  end team
  change team (team, ca[*] => selector)
  end team
  change team (team, ca[*] => selector, ca2[2,*] => selector2d)
  end team
  end block

  !A selector may appear only once in selector-list.
  ! ERROR: Selector 'selector' was already used as a selector or coarray in this statement
  change team (team, ca[*] => selector, ca2[*] => selector)
  end team

  ! Within a CHANGE TEAM construct, a CYCLE or EXIT statement is not allowed if it belongs to an outer construct.
  block
  outer1: if (.true.) then
    change team (team)
      if (.true.) then
        ! ERROR: EXIT must not leave a CHANGE TEAM statement
        exit outer1
      end if
    end team
  end if outer1
  end block
  block
  outer2: do
    change team (team)
      ! ERROR: CYCLE must not leave a CHANGE TEAM statement
      cycle outer2
    end team
  end do outer2
  end block

  ! The construct name must not be the same as any other construct name in the scoping unit.
  block
  construct2: block
  end block construct2
  ! ERROR: 'construct2' is already declared in this scoping unit
  construct2: change team (team)
  end team construct2
  end block

  ! When the CHANGE TEAM statement is executed, the selectors must all be established coarrays.
  ! ERROR: Selector in coarray association must name a coarray
  change team (team, ca[*] => not_selector)
  end team

  ! The coarray name in a coarray-association must not be the same as the name as the name of another coarray or of a selector in the CHANGE TEAM statement.
  ! ERROR: 'selector' is not an object that can appear in an expression
  change team (team, selector[*] => selector)
  end team
end subroutine
