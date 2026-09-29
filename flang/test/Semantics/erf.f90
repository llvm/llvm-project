! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=ERROR %s

function derf8_error4(x)
  real(kind=8) :: derf8_error4
  real(kind=4) :: x
  derf8_error4 = derf(x);
! ERROR: Actual argument for 'x=' has bad type or kind 'REAL(4)'
end function derf8_error4

function derf8_error16(x)
  real(kind=8) :: derf8_error16
  real(kind=16) :: x
  derf8_error16 = derf(x);
! ERROR: Actual argument for 'x=' has bad type or kind 'REAL(16)'
end function derf8_error16

function qerf16_error4(x)
  real(kind=16) :: qerf16_error4
  real(kind=4) :: x
  qerf16_error4 = qerf(x);
! ERROR: Actual argument for 'x=' has bad type or kind 'REAL(4)'
end function qerf16_error4

function qerf16_error8(x)
  real(kind=16) :: qerf16_error8
  real(kind=8) :: x
  qerf16_error8 = qerf(x);
! ERROR: Actual argument for 'x=' has bad type or kind 'REAL(8)'
end function qerf16_error8
