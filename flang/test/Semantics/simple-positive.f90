! Test accepted SIMPLE procedure cases.
! RUN: %flang_fc1 -fsyntax-only %s

! SIMPLE procedures satisfy PURE requirements.
module simple_satisfies_pure
  implicit none
  abstract interface
    pure integer function pure_iface(x)
      integer, intent(in) :: x
    end function
  end interface
contains
  simple integer function simple_impl(x)
    integer, intent(in) :: x
    simple_impl = x
  end function

  pure integer function apply_requires_pure(f, x)
    procedure(pure_iface) :: f
    integer, intent(in) :: x
    apply_requires_pure = f(x)
  end function

  integer function test()
    test = apply_requires_pure(simple_impl, 1)
  end function
end module

! Use-associated SIMPLE procedures are recognized.
module simple_use_assoc_m
contains
  simple subroutine s()
  end subroutine
end module

program simple_use_assoc_p
  use simple_use_assoc_m
  implicit none
  call needs_simple(s)
contains
  subroutine needs_simple(p)
    abstract interface
      simple subroutine simple_proc()
      end subroutine
    end interface
    procedure(simple_proc) :: p
  end subroutine
end program

! SIMPLE functions returning non-trivial CHARACTER results.
module simple_character_result
contains
  simple function fixed_len_character() result(res)
    character(len=5) :: res
    res = "hello"
  end function

  simple function dependent_len_character(x) result(res)
    character(*), intent(in) :: x
    character(len=len(x)) :: res
    res = x
  end function

  subroutine test()
    character(len=3) :: a
    character(len=7) :: b
    a = dependent_len_character("abc")
    b = dependent_len_character("testing")
  end subroutine
end module

! SIMPLE and PURE may both appear in either order.
module simple_pure_order
contains
  simple pure subroutine simple_pure_sub()
  end subroutine

  pure simple subroutine pure_simple_sub()
  end subroutine

  subroutine test()
    call simple_pure_sub()
    call pure_simple_sub()
  end subroutine
end module

! SIMPLE separate module function is accepted.
module m_smf
  interface
    simple module function f(x) result(res)
      integer, intent(in) :: x
      integer :: res
    end function
  end interface
end module

submodule(m_smf) sm_smf
contains
  module procedure f
    res = x
  end procedure
end submodule

! MODULE SIMPLE function prefix order is accepted.
module m_msf
  interface
    module simple function g(x) result(res)
      integer, intent(in) :: x
      integer :: res
    end function
  end interface
end module

submodule(m_msf) sm_msf
contains
  module procedure g
    res = x
  end procedure
end submodule
