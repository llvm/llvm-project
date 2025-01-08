! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine s()
  real(8) :: a
  !ERROR: COMPLEX(KIND=128) is not a supported type
  complex(128) :: x
  a(i)=a + ((i)+1) + 3.14
  !ERROR: 'a' has not been declared as an array or pointer-valued function
  a()=z(a * a + n-1 - x) + i((/0,0,0,0,0,0,0,0,0,0/)) + 8
end
