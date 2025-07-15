! RUN: %python %S/test_errors.py %s %flang_fc1
bind(c) :: /blk/
!ERROR: 'x' may not be a member of BIND(C) COMMON block /blk/
common /blk/ x
!BECAUSE: A scalar interoperable variable may not be ALLOCATABLE or POINTER
integer, pointer :: x
end
