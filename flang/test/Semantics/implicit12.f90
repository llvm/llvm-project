! RUN: %python %S/test_errors.py %s %flang_fc1
use iso_c_binding, only: c_ptr, c_associated
implicit none(external)
type (c_ptr) :: cptr
if (.not. c_associated (cptr)) then
   return
end if
end
