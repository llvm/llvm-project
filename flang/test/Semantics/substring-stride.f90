! RUN: %python %S/test_errors.py %s %flang_fc1

subroutine substring_stride(s, a)
  character(8) :: s, a(8)
  type box
    character(8) :: field
    character(8) :: array(8)
  end type
  type(box) :: b(2)

  !ERROR: A substring cannot have a stride
  print *, s(::2)
  !ERROR: A substring cannot have a stride
  print *, s(1::2)
  !ERROR: A substring cannot have a stride
  print *, s(:8:2)
  !ERROR: A substring cannot have a stride
  print *, s(1:8:2)
  !ERROR: A substring cannot have a stride
  print *, b(1)%field(1:8:2)
  !ERROR: A substring cannot have a stride
  print *, b(:)%field(1:8:2)

  ! Valid references and strided character array sections remain valid.
  print *, s, s(1:4), b(1)%field, b(:)%field(1:4)
  print *, a(::2), b(1)%array(::2)
end subroutine
