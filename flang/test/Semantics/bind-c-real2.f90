! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! REAL(KIND=2)/COMPLEX(KIND=2) (IEEE binary16) are interoperable with C
! _Float16, and ISO_C_BINDING exports c_float16/c_float16_complex (value 2).
! Verify no diagnostic (including no portability warning under -pedantic),
! while bfloat16 (KIND=3) stays non-interoperable.
module m
  use iso_c_binding, only: c_float16, c_float16_complex
  integer, parameter :: check = c_float16
  real(kind=2), bind(c) :: a
  complex(kind=2), bind(c) :: b
  real(c_float16), bind(c) :: c
  complex(c_float16_complex), bind(c) :: d

  type, bind(c) :: t1
    real(kind=2) :: x
    complex(kind=2) :: y
  end type

  type, bind(c) :: t2
    !ERROR: Each component of an interoperable derived type must have an interoperable type
    real(kind=3) :: x ! bfloat16 is not interoperable
  end type

  interface
    subroutine s(x, y) bind(c)
      import :: c_float16, c_float16_complex
      real(c_float16), value :: x
      complex(c_float16_complex), value :: y
    end subroutine
  end interface
end
