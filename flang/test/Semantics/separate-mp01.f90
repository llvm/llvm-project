! RUN: %S/test_errors.sh %s %t %f18

! case 1: ma_create_new_fun' was not declared a separate module procedure
module m1
  integer :: i
  interface ma
    module function ma_create_fun( ) result(this)
      integer this
    end function
  end interface
end module

submodule (m1) ma_submodule
  integer :: j
  contains
  module function ma_create_fun() result(this)
    integer this
    i = 1
    j = 2
  end function

  !ERROR: 'ma_create_new_fun' was not declared a separate module procedure
  module function ma_create_new_fun() result(this)
    integer :: this
    i = 2
    j = 1
    print *, "Hello"
  end function
end submodule

! case 2: 'mb_create_new_sub' was not declared a separate module procedure
module m2
  integer :: i
  interface mb
    module subroutine  mb_create_sub
    end subroutine mb_create_sub
  end interface
end module

submodule (m2) mb_submodule
  integer :: j
  contains
  module subroutine  mb_create_sub
    integer this
    i = 1
    j = 2
  end subroutine mb_create_sub

  !ERROR: 'mb_create_new_sub' was not declared a separate module procedure
  module SUBROUTINE  mb_create_new_sub() 
    integer :: this
    i = 2
    j = 1
  end SUBROUTINE mb_create_new_sub
end submodule

! case 3: separate module procedure without module prefix
module m3
  interface mc
    function mc_create( ) result(this)
      integer :: this
    end function
  end interface
end module

submodule (m3) mc_submodule
  contains
  !ERROR: 'mc_create' was not declared a separate module procedure
  module function mc_create() result(this)
    integer :: this
  end function
end submodule

! case 4: Submodule having separate module procedure rather than a module
module m4
  interface
    real module function func1()   ! module procedure interface body for func1
    end function
  end interface
end module

submodule (m4) m4sub
  interface
    module function func2(b)       ! module procedure interface body for func2
      integer :: b
      integer :: func2
    end function

    real module function func3()   ! module procedure interface body for func3
    end function
  end interface
  contains
    real module function func1()   ! implementation of func1 declared in m4
      func1 = 20
    end function
end submodule

submodule (m4:m4sub) m4sub2
  contains
    module function func2(b)       ! implementation of func2 declared in m4sub
      integer :: b
      integer :: func2
      func2 = b
    end function

    real module function func3()   ! implementation of func3 declared in m4sub
      func3 = 20
    end function
end submodule
