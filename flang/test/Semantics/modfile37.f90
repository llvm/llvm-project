! RUN: %S/test_modfile.sh %s %t %f18

! Ensure that a dummy procedure's interface's imports
! appear in the module file.

module m
  type :: t
  end type
 contains
  subroutine s1(s2)
    interface
      subroutine s2(x)
        import
        class(t) :: x
      end subroutine
    end interface
  end subroutine
end module
!Expect: m.mod
!module m
!type::t
!end type
!contains
!subroutine s1(s2)
!interface
!subroutine s2(x)
!import::t
!class(t)::x
!end
!end interface
!end
!end
