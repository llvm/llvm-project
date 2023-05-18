! RUN: %python %S/test_errors.py %s %flang_fc1
module m1
  type t1
   contains
    procedure :: tbp => s1
  end type
  type, extends(t1) :: t1e
   contains
    !ERROR: A type-bound procedure and its override must have compatible interfaces
    procedure :: tbp => s1e
  end type
 contains
  subroutine s1(x)
    class(t1) :: x
  end
  subroutine s1e(x)
    class(t1e), intent(in out) :: x
  end
end

module m2
  type t1
   contains
    procedure :: tbp => s1
  end type
  type, extends(t1) :: t1e
   contains
    !ERROR: A type-bound procedure and its override must have compatible interfaces
    procedure :: tbp => s1e
  end type
 contains
  subroutine s1(x)
    class(t1), intent(in out) :: x
  end
  subroutine s1e(x)
    class(t1e) :: x
  end
end

module m3
  type t1
   contains
    procedure, nopass :: tbp => s1
  end type
  type, extends(t1) :: t1e
   contains
   !ERROR: A NOPASS type-bound procedure and its override must have identical interfaces
    procedure, nopass :: tbp => s1e
  end type
 contains
  subroutine s1(x)
    real, intent(in out) :: x
  end
  subroutine s1e(x)
    real :: x
  end
end
