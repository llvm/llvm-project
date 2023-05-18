! RUN: %python %S/test_errors.py %s %flang_fc1
integer :: g(10)
f(i) = i + 1  ! statement function
g(i) = i + 2  ! mis-parsed assignment
!ERROR: 'h' has not been declared as an array or pointer-valued function
h(i) = i + 3
end
