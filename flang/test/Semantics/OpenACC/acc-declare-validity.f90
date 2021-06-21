! RUN: %S/../test_errors.sh %s %t %flang -fopenacc
! REQUIRES: shell

! Check OpenACC clause validity for the following construct and directive:
!   2.13 Declare

module openacc_declare_validity

  implicit none

  real(8), dimension(10) :: aa, bb, ab, cc

  !ERROR: At least one clause is required on the DECLARE directive
  !$acc declare

  !$acc declare create(aa, bb)

  !$acc declare link(ab)

  !$acc declare device_resident(cc)

  !ERROR: COPYOUT clause is not allowed on the DECLARE directive in module declaration section
  !$acc declare copyout(ab)

  !ERROR: COPY clause is not allowed on the DECLARE directive in module declaration section
  !$acc declare copy(ab)

  !ERROR: PRESENT clause is not allowed on the DECLARE directive in module declaration section
  !$acc declare present(ab)

  !ERROR: DEVICEPTR clause is not allowed on the DECLARE directive in module declaration section
  !$acc declare deviceptr(ab)

contains

  subroutine sub1(cc, dd)
    real(8) :: cc(:)
    real(8) :: dd(:)
    !$acc declare present(cc, dd)
  end subroutine sub1

  function fct1(ee, ff, gg, hh, ii)
    integer :: fct1
    real(8), intent(in) :: ee(:)
    !$acc declare copyin(readonly: ee)
    real(8) :: ff(:), hh(:), ii(:,:)
    !$acc declare link(hh) device_resident(ii)
    real(8), intent(out) :: gg(:)
    !$acc declare copy(ff) copyout(gg)
  end function fct1

  subroutine sub2(cc)
    real(8), dimension(*) :: cc
    !ERROR: Assumed-size dummy arrays may not appear on the DECLARE directive
    !$acc declare present(cc)
  end subroutine sub2

end module openacc_declare_validity
