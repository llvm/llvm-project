! RUN: %python %S/test_errors.py %s %flang_fc1

module protected_target_override
  type base
  contains
    procedure :: s => bs
  end type
  type, extends(base) :: ext
  contains
    procedure :: s => es
  end type
  !ERROR: Generic 'g' may not have specific procedures 'g1' and 'g2' as their interfaces are not distinguishable
  interface g
    module procedure g1, g2
  end interface
contains
  subroutine bs(this, a)
    class(base) :: this
    integer, pointer, protected_target :: a
  end
  subroutine es(this, a)
    class(ext) :: this
    integer, pointer :: a
  end
  subroutine g1(a)
    integer, pointer, protected_target :: a
  end
  subroutine g2(a)
    integer, pointer :: a
  end
end
