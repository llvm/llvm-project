! RUN: %python %S/test_errors.py %s %flang_fc1

module protected_target_function_result
  integer, target, save :: x = 1
  integer, target, save :: y(2) = [1, 2]
contains
  function f() result(r)
    integer, pointer, protected_target :: r
    r => x
  end
  function g() result(r)
    class(*), pointer, protected_target :: r(:)
    r => y
  end
  subroutine s2(d)
    integer, intent(inout) :: d
  end
  subroutine t
    type box
      integer, pointer :: p
    end type
    type(box) :: h
    integer, pointer :: q
    integer, pointer, protected_target :: pq
    external ext
    !ERROR: Pointer assignment target is a PROTECTED_TARGET pointer or a subobject of one; the left-hand side pointer must also have the PROTECTED_TARGET attribute
    q => f()
    pq => f()
    !ERROR: Structure constructor component 'p' must have the PROTECTED_TARGET attribute when its pointer target is a PROTECTED_TARGET pointer or a subobject of one
    h = box(f())
    !ERROR: Actual argument that is a PROTECTED_TARGET pointer or a subobject of one may not be passed to a procedure with an implicit interface
    call ext(f())
    !ERROR: Non-POINTER dummy argument 'd=' must have INTENT(IN) when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
    call s2(f())
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'f()' may not be defined via PROTECTED_TARGET pointer function result
    f() = 3
    select type (a => g())
    type is (integer)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'g()' may not be defined via PROTECTED_TARGET pointer function result
      a(1) = 4
      !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'd=' is not definable
      !BECAUSE: 'g()' may not be defined via PROTECTED_TARGET pointer function result
      call s2(a(1))
    end select
  end
end
