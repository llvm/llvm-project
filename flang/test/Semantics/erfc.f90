! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=ERROR %s

function derfc8_error4(x)
  real(kind=8) :: derfc8_error4
  real(kind=4) :: x
  derfc8_error4 = derfc(x);
! ERROR: Actual argument for 'x=' has bad type or kind 'REAL(4)'
end function derfc8_error4

function derfc8_error16(x)
  real(kind=8) :: derfc8_error16
  real(kind=16) :: x
  derfc8_error16 = derfc(x);
! ERROR: Actual argument for 'x=' has bad type or kind 'REAL(16)'
end function derfc8_error16

function qerfc16_error4(x)
  real(kind=16) :: qerfc16_error4
  real(kind=4) :: x
  qerfc16_error4 = qerfc(x);
! ERROR: Actual argument for 'x=' has bad type or kind 'REAL(4)'
end function qerfc16_error4

function qerfc16_error8(x)
  real(kind=16) :: qerfc16_error8
  real(kind=8) :: x
  qerfc16_error8 = qerfc(x);
! ERROR: Actual argument for 'x=' has bad type or kind 'REAL(8)'
end function qerfc16_error8
