! RUN: %python %S/test_errors.py %s %flang_fc1
! C729 A derived type type-name shall not be DOUBLEPRECISION or the same as 
! the name of any intrinsic type defined in this document.
subroutine s()
  ! This one's OK
  type derived
  end type
  !ERROR: A derived type name cannot be the name of an intrinsic type
  type integer
  end type
  !ERROR: A derived type name cannot be the name of an intrinsic type
  type real
  end type
  !ERROR: A derived type name cannot be the name of an intrinsic type
  type doubleprecision
  end type
  !ERROR: A derived type name cannot be the name of an intrinsic type
  type complex
  end type
  !ERROR: A derived type name cannot be the name of an intrinsic type
  type character
  end type
  !ERROR: A derived type name cannot be the name of an intrinsic type
  type logical
  end type
end subroutine s
