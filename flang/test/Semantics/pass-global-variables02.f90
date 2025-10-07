!RUN: %python %S/test_errors.py %s %flang_fc1 -Werror -Wpass-global-variable
module test_mod
  implicit none (type, external)

  type :: wt
    integer :: ival
  end type wt
  type :: qt
    type(wt) :: w
  end type qt
  type(wt) :: w(2)
  type(qt) :: q

  integer, parameter :: ipar = 1
  integer, private :: ipri
  integer, public ::  ipub

  common /ex/ ic
  integer :: ic

contains
  subroutine pass_int_in(i)
    integer, intent(in) :: i
  end subroutine pass_int_in
  subroutine pass_int(i)
    integer, intent(inout) :: i
  end subroutine pass_int
  pure subroutine pure_int(i)
    integer, intent(inout) :: i
  end subroutine pure_int
  subroutine pass_ival(i)
    integer, value :: i
  end subroutine pass_ival
  subroutine pass_qt(q)
    type(qt), intent(in) :: q
  end subroutine pass_qt

  subroutine tests()

    call pass_ival(ipub)       !< ok:      pass to value
    call pass_int_in(ipar)     !< ok:      pass parameter
    call pass_int(ipri)        !< ok:      pass private
    !WARNING: Passing global variable 'ipub' from MODULE 'test_mod' as function argument [-Wpass-global-variable]
    call pass_int(ipub)        !< warn:    pass public
    call pure_int(ipub)        !< ok:      pass to pure
    call pass_int(w(1)%ival)   !< ok:      comes from derived
    call pass_qt(q)            !< ok:      derived

    ipub = iand(ipub, ipar)    !< ok:      passed to intrinsic

    call pass_ival(ic)         !< ok:      passed to value
    !WARNING: Passing global variable 'ic' from COMMON 'ex' as function argument [-Wpass-global-variable]
    call pass_int_in(ic)       !< warn:    intent(in) does not guarantee that ic is not changing during call
    !WARNING: Passing global variable 'ic' from COMMON 'ex' as function argument [-Wpass-global-variable]
    call pass_int(ic)          !< warn:    global variable may be changed during call
    call pure_int(ic)          !< ok:      pure keeps value

    ic = iand(ic, ic)          !< ok:      passed to intrinsic

  end subroutine tests
end module test_mod
