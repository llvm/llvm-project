! RUN: %S/test_errors.sh %s %t %f18
module m
!C778 The same binding-attr shall not appear more than once in a given
!binding-attr-list.
!
!R749 type-bound-procedure-stmt
!  PROCEDURE [ [ ,binding-attr-list] :: ]type-bound-proc-decl-list
!  or PROCEDURE (interface-name),binding-attr-list::binding-name-list
!
!
!  binding-attr values are:
!    PUBLIC, PRIVATE, DEFERRED, NON_OVERRIDABLE, NOPASS, PASS [ (arg-name) ]
!
  type, abstract :: boundProcType
   contains
    !WARNING: Attribute 'PUBLIC' cannot be used more than once
    procedure(subPublic), public, deferred, public :: publicBinding
    !WARNING: Attribute 'PRIVATE' cannot be used more than once
    procedure(subPrivate), private, deferred, private :: privateBinding
    !WARNING: Attribute 'DEFERRED' cannot be used more than once
    procedure(subDeferred), deferred, public, deferred :: deferredBinding
    !WARNING: Attribute 'NON_OVERRIDABLE' cannot be used more than once
    procedure, non_overridable, public, non_overridable :: subNon_overridable;
    !WARNING: Attribute 'NOPASS' cannot be used more than once
    procedure(subNopass), nopass, deferred, nopass :: nopassBinding
    !WARNING: Attribute 'PASS' cannot be used more than once
    procedure(subPass), pass, deferred, pass :: passBinding
    !ERROR: Attributes 'PASS' and 'NOPASS' conflict with each other
    procedure(subPassNopass), pass, deferred, nopass :: passNopassBinding  ! C781
  end type boundProcType

contains
    subroutine subPublic(x)
      class(boundProcType), intent(in) :: x
    end subroutine subPublic

    subroutine subPrivate(x)
      class(boundProcType), intent(in) :: x
    end subroutine subPrivate

    subroutine subDeferred(x)
      class(boundProcType), intent(in) :: x
    end subroutine subDeferred

    subroutine subNon_overridable(x)
      class(boundProcType), intent(in) :: x
    end subroutine subNon_overridable

    subroutine subNopass(x)
      class(boundProcType), intent(in) :: x
    end subroutine subNopass

    subroutine subPass(x)
      class(boundProcType), intent(in) :: x
    end subroutine subPass

    subroutine subPassNopass(x)
      class(boundProcType), intent(in) :: x
    end subroutine subPassNopass

end module m
