! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=ERROR %s

function derfc_scaled8_error4(x)
  real(kind=8) :: derfc_scaled8_error4
  real(kind=4) :: x
  derfc_scaled8_error4 = derfc_scaled(x);
! ERROR: Actual argument for 'x=' has bad type or kind 'REAL(4)'
end function derfc_scaled8_error4

function derfc_scaled8_error16(x)
  real(kind=8) :: derfc_scaled8_error16
  real(kind=16) :: x
  derfc_scaled8_error16 = derfc_scaled(x);
! ERROR: Actual argument for 'x=' has bad type or kind 'REAL(16)'
end function derfc_scaled8_error16

function qerfc_scaled16_error4(x)
  real(kind=16) :: qerfc_scaled16_error4
  real(kind=4) :: x
  qerfc_scaled16_error4 = qerfc_scaled(x);
! ERROR: Actual argument for 'x=' has bad type or kind 'REAL(4)'
end function qerfc_scaled16_error4

function qerfc_scaled16_error8(x)
  real(kind=16) :: qerfc_scaled16_error8
  real(kind=8) :: x
  qerfc_scaled16_error8 = qerfc_scaled(x);
! ERROR: Actual argument for 'x=' has bad type or kind 'REAL(8)'
end function qerfc_scaled16_error8
