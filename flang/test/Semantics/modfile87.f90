! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m.f90
! RUN: %flang_fc1 -fsyntax-only -pedantic -J%t %t/s.f90 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fsyntax-only -J%t %t/protected-target.f90
! RUN: FileCheck --check-prefix=MOD-PT %s < %t/protected_target_mod.mod
! RUN: %flang_fc1 -fsyntax-only -J%t %t/protected-target-use.f90
! RUN: %python %S/test_errors.py %t/protected-target-errors.f90 %flang_fc1 -J%t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/protected-target-interfaces.f90
! RUN: %python %S/test_errors.py %t/protected-target-separate.f90 %flang_fc1 -J%t

! When a module and its submodule are compiled in separate
! invocations (so that the submodule's parent module scope is read back from
! the .mod file), the "missing MODULE prefix" portability warning for the
! submodule's subprogram must still be emitted.

!--- m.f90
module modfile87m
  interface
    module subroutine inside_one()
    end subroutine
  end interface
end module

!--- s.f90
submodule (modfile87m) modfile87s
contains
  subroutine inside_one()
  end subroutine
end submodule

!CHECK: portability: Subprogram 'inside_one' in this submodule is missing the MODULE prefix

!--- protected-target.f90
module protected_target_mod
  integer, target :: value
  integer, target, protected :: protected_value
  integer, pointer, protected_target :: p
  type t
    integer, pointer, protected_target :: component
  end type
contains
  function get() result(r)
    integer, pointer, protected_target :: r
    r => value
  end function
  subroutine consume(x)
    integer, pointer, protected_target, intent(in) :: x
  end subroutine
end module

! MOD-PT-DAG: integer(4),pointer,protected_target::p
! MOD-PT-DAG: integer(4),pointer,protected_target::component
! MOD-PT-DAG: integer(4),pointer,protected_target::r
! MOD-PT-DAG: integer(4),intent(in),pointer,protected_target::x

!--- protected-target-use.f90
subroutine use_protected_target
  use protected_target_mod, only: p, get, consume, t, protected_value
  integer, pointer, protected_target :: keep
  type(t) :: object
  procedure(get), pointer :: get_pointer
  procedure(consume), pointer :: consume_pointer
  keep => p
  object = t(p)
  keep => get()
  get_pointer => get
  consume_pointer => consume
  call consume(p)
  call consume(object%component)
  call consume(protected_value)
  print *, p
  nullify(keep)
end subroutine

!--- protected-target-errors.f90
subroutine imported_protected_target
  use protected_target_mod, only: imported => p, get, consume, t
  integer, pointer :: ordinary
  type(t) :: object
  abstract interface
    function ordinary_result() result(r)
      integer, pointer :: r
    end function
    subroutine ordinary_dummy(x)
      integer, pointer, intent(in) :: x
    end subroutine
  end interface
  procedure(ordinary_result), pointer :: result_pointer
  procedure(ordinary_dummy), pointer :: dummy_pointer
  !ERROR: A PROTECTED_TARGET target may not be associated with pointer 'ordinary' that is not PROTECTED_TARGET
  ordinary => imported
  !ERROR: A PROTECTED_TARGET target may not be associated with pointer 'ordinary' that is not PROTECTED_TARGET
  ordinary => get()
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer
  imported = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'component' is a PROTECTED_TARGET pointer
  object%component = 1
  !ERROR: Object in DEALLOCATE statement is not deallocatable
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer
  deallocate(imported)
  ! PROTECTED_TARGET is not a procedure characteristic.
  result_pointer => get
  dummy_pointer => consume
end subroutine

!--- protected-target-interfaces.f90
module protected_target_interfaces
  abstract interface
    function ordinary_result() result(r)
      integer, pointer :: r
    end function
    function protected_result() result(r)
      integer, pointer, protected_target :: r
    end function
    subroutine ordinary_dummy(x)
      integer, pointer, intent(in) :: x
    end subroutine
    subroutine protected_dummy(x)
      integer, pointer, protected_target, intent(in) :: x
    end subroutine
  end interface
  interface
    module subroutine ordinary_dummy_body(x)
      integer, pointer, intent(in) :: x
    end subroutine
    module subroutine protected_dummy_body(x)
      integer, pointer, protected_target, intent(in) :: x
    end subroutine
    module function ordinary_result_body() result(r)
      integer, pointer :: r
    end function
    module function protected_result_body() result(r)
      integer, pointer, protected_target :: r
    end function
    module subroutine ordinary_nested_body(f)
      procedure(ordinary_result) :: f
    end subroutine
    module subroutine protected_nested_body(f)
      procedure(protected_result) :: f
    end subroutine
    module subroutine ordinary_nested_dummy_body(s)
      procedure(ordinary_dummy) :: s
    end subroutine
    module subroutine protected_nested_dummy_body(s)
      procedure(protected_dummy) :: s
    end subroutine
  end interface
end module

!--- protected-target-separate.f90
! PROTECTED_TARGET is not a procedure characteristic, so these definitions
! match the interface bodies read from the module file.
submodule(protected_target_interfaces) protected_target_separate
contains
  module subroutine ordinary_dummy_body(x)
    integer, pointer, protected_target, intent(in) :: x
  end subroutine
  module subroutine protected_dummy_body(x)
    integer, pointer, intent(in) :: x
  end subroutine
  module function ordinary_result_body() result(r)
    integer, pointer, protected_target :: r
    r => null()
  end function
  module function protected_result_body() result(r)
    integer, pointer :: r
    r => null()
  end function
  module subroutine ordinary_nested_body(f)
    procedure(protected_result) :: f
  end subroutine
  module subroutine protected_nested_body(f)
    procedure(ordinary_result) :: f
  end subroutine
  module subroutine ordinary_nested_dummy_body(s)
    procedure(protected_dummy) :: s
  end subroutine
  module subroutine protected_nested_dummy_body(s)
    procedure(ordinary_dummy) :: s
  end subroutine
end submodule
