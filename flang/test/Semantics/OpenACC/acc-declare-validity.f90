! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.13 Declare

module openacc_declare_validity

  implicit none

  real(8), dimension(10) :: aa, bb, ab, ac, ad, ae, af, cc, dd

  !ERROR: At least one clause is required on the DECLARE directive
  !$acc declare

  !$acc declare create(aa, bb)

  !ERROR: 'aa' in the CREATE clause is already present in another clause in this module
  !$acc declare create(aa)

  !$acc declare link(ab)

  !$acc declare device_resident(cc)

  !ERROR: COPYOUT clause is not allowed on the DECLARE directive in module declaration section
  !$acc declare copyout(ac)

  !ERROR: COPY clause is not allowed on the DECLARE directive in module declaration section
  !$acc declare copy(af)

  !ERROR: PRESENT clause is not allowed on the DECLARE directive in module declaration section
  !$acc declare present(ad)

  !ERROR: DEVICEPTR clause is not allowed on the DECLARE directive in module declaration section
  !$acc declare deviceptr(ae)

  !ERROR: The ZERO modifier is not allowed for the CREATE clause on the DECLARE directive
  !$acc declare create(zero: dd)

contains

  subroutine sub1(cc, dd)
    real(8) :: cc(:)
    real(8) :: dd(:)
    !$acc declare present(cc, dd)
    !ERROR: 'cc' in the CREATE clause is already present in another clause in this module
    !$acc declare create(cc)
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

  subroutine sub3()
    real :: aa(100)
    !ERROR: The ZERO modifier is not allowed for the COPYOUT clause on the DECLARE directive
    !$acc declare copyout(zero: aa)
  end subroutine

end module openacc_declare_validity
