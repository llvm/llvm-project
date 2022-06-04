! RUN: %python %S/test_modfile.py %s %flang_fc1

! Check modfile that contains import of use-assocation of another use-association.

module m1
  interface
     subroutine s(x)
       use, intrinsic :: iso_c_binding, only: c_ptr
       type(c_ptr) :: x
     end subroutine
  end interface
end module
!Expect: m1.mod
!module m1
! interface
!  subroutine s(x)
!   use,intrinsic::iso_c_binding, only: c_ptr
!   type(c_ptr) :: x
!  end
! end interface
!end

module m2
  use, intrinsic :: iso_c_binding, only: c_ptr
  interface
     subroutine s(x)
       import :: c_ptr
       type(c_ptr) :: x
     end subroutine
  end interface
end module
!Expect: m2.mod
!module m2
! use,intrinsic::iso_c_binding,only:c_ptr
! interface
!  subroutine s(x)
!   import::c_ptr
!   type(c_ptr)::x
!  end
! end interface
!end
