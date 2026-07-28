! RUN: %python %S/test_errors.py %s %flang_fc1

module m
  type, abstract :: parent
   contains
    procedure(ip), deferred :: set
  end type
  abstract interface
    subroutine ip(x)
      import parent
      class(parent) :: x
    end subroutine
  end interface
  type, public, extends(parent), abstract :: child
  end type
  type, extends(child) :: grandchild
   contains
    !ERROR: Dummy arguments of type-bound procedure 'set' and its override must correspond by name and position
    procedure :: set
  end type
 contains
  subroutine set(y)
    class(grandchild) :: y
  end subroutine
end module

! Valid override with matching dummy names
module m_ok
  type, abstract :: t
   contains
    procedure(ip), deferred :: set
  end type
  abstract interface
    subroutine ip(x)
      import t
      class(t) :: x
    end subroutine
  end interface
  type, extends(t) :: e
   contains
    procedure :: set
  end type
 contains
  subroutine set(x)
    class(e) :: x
  end subroutine
end module

! Extra dummy argument: interface mismatch, not a name error
module m_extra
  type, abstract :: t
   contains
    procedure(ip), deferred :: set
  end type
  abstract interface
    subroutine ip(x)
      import t
      class(t) :: x
    end subroutine
  end interface
  type, extends(t) :: e
   contains
    !ERROR: A type-bound procedure and its override must have compatible interfaces
    procedure :: set
  end type
 contains
  subroutine set(x, n)
    class(e) :: x
    integer :: n
  end subroutine
end module

! Non-pass dummy renamed
module m_rename
  type, abstract :: t
   contains
    procedure(ip), deferred :: set
  end type
  abstract interface
    subroutine ip(x, n)
      import t
      class(t) :: x
      integer :: n
    end subroutine
  end interface
  type, extends(t) :: e
   contains
    !ERROR: Dummy arguments of type-bound procedure 'set' and its override must correspond by name and position
    procedure :: set
  end type
 contains
  subroutine set(x, m2)
    class(e) :: x
    integer :: m2
  end subroutine
end module

! NOPASS deferred override with mismatched dummy name
module m_nopass
  type, abstract :: t
   contains
    procedure(ip), deferred, nopass :: act
  end type
  abstract interface
    subroutine ip(a)
      integer :: a
    end subroutine
  end interface
  type, extends(t) :: e
   contains
    !ERROR: Dummy arguments of type-bound procedure 'act' and its override must correspond by name and position
    procedure, nopass :: act => impl
  end type
 contains
  subroutine impl(b)
    integer :: b
  end subroutine
end module
