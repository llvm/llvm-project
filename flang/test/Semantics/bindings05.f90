! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
module m1
  type base
   contains
    procedure, private :: binding => basesub
    generic :: generic => binding
  end type
  type, extends(base) :: ext1
   contains
    procedure, private :: binding => ext1sub
  end type
 contains
  subroutine basesub(x)
    class(base), intent(in) :: x
  end
  subroutine ext1sub(x)
    class(ext1), intent(in) :: x
  end
  subroutine test1
    type(ext1) x
!CHECK: CALL ext1sub(x)
    call x%generic
  end
end

module m2
  use m1
  type, extends(ext1) :: ext2
   contains
    procedure :: binding => ext2sub
  end type
 contains
  subroutine ext2sub(x)
    class(ext2), intent(in) :: x
  end
  subroutine test2
    type(ext2) x
!CHECK: CALL ext1sub(x)
    call x%generic ! private binding not overridable
  end
end

module m3
  type base
   contains
    procedure, public :: binding => basesub
    generic :: generic => binding
  end type
  type, extends(base) :: ext1
   contains
    procedure, public :: binding => ext1sub
  end type
 contains
  subroutine basesub(x)
    class(base), intent(in) :: x
  end
  subroutine ext1sub(x)
    class(ext1), intent(in) :: x
  end
  subroutine test1
    type(ext1) x
!CHECK: CALL ext1sub(x)
    call x%generic
  end
end

module m4
  use m3
  type, extends(ext1) :: ext2
   contains
    procedure :: binding => ext2sub
  end type
 contains
  subroutine ext2sub(x)
    class(ext2), intent(in) :: x
  end
  subroutine test2
    type(ext2) x
!CHECK: CALL ext2sub(x)
    call x%generic ! public binding is overridable
  end
end

module m5
  type base
   contains
    procedure, private :: binding => basesub
    generic :: generic => binding
  end type
  type, extends(base) :: ext1
   contains
    procedure, public :: binding => ext1sub
  end type
 contains
  subroutine basesub(x)
    class(base), intent(in) :: x
  end
  subroutine ext1sub(x)
    class(ext1), intent(in) :: x
  end
  subroutine test1
    type(ext1) x
!CHECK: CALL ext1sub(x)
    call x%generic
  end
end

module m6
  use m5
  type, extends(ext1) :: ext2
   contains
    procedure :: binding => ext2sub
  end type
 contains
  subroutine ext2sub(x)
    class(ext2), intent(in) :: x
  end
  subroutine test2
    type(ext2) x
!CHECK: CALL ext2sub(x)
    call x%generic ! public binding is overridable
  end
end
