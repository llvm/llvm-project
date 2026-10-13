! RUN: %python %S/test_errors.py %s %flang_fc1
! PROTECTED_TARGET attribute (F2028 draft J3/26-007r1): pointer association,
! structure constructors, and actual arguments (C870-C874)
module m
  type t
    real :: v
    real, pointer :: q
    real, pointer, protected_target :: pq
  end type
  type ta
    real, allocatable :: a(:)
  end type
  interface gen
    subroutine genReal(a)
      real, pointer :: a
    end
    subroutine genInteger(n)
      integer, pointer :: n
    end
  end interface
  interface
    subroutine plainPointer(a)
      real, pointer :: a
    end
    subroutine ptPointer(a)
      real, pointer, protected_target :: a
    end
    subroutine inPointer(a)
      real, pointer, intent(in) :: a(:)
    end
    subroutine ptInPointer(a)
      real, pointer, protected_target, intent(in) :: a(:)
    end
    subroutine ptOutPointer(a)
      real, pointer, protected_target, intent(out) :: a
    end
    subroutine noIntent(a)
      real :: a
    end
    subroutine inIntent(a)
      real, intent(in) :: a
    end
    subroutine outIntent(a)
      real, intent(out) :: a
    end
    subroutine valueArg(a)
      real, value :: a
    end
    subroutine allocNoIntent(a)
      real, allocatable :: a(:)
    end
    subroutine allocIn(a)
      real, allocatable, intent(in) :: a(:)
    end
    subroutine allocTkr(a)
      !dir$ ignore_tkr(p) a
      real, allocatable :: a(:)
    end
    function ptResult()
      real, pointer, protected_target :: ptResult
    end
    function plainResult()
      real, pointer :: plainResult
    end
  end interface
  interface
    module subroutine separate(a)
      real, pointer, protected_target :: a
    end
  end interface
 contains
  subroutine association(x, xs)
    real, target :: x, xs(:)
    real, pointer :: w, ws(:)
    real, pointer, protected_target :: p, ps(:), p2
    type(t) :: tt
    type(t), pointer, protected_target :: pt
    p => x
    ps => xs
    p2 => p
    p2 => w
    p2 => ptResult()
    p2 => plainResult()
    w => pt%q
    !ERROR: Pointer 'w' must have the PROTECTED_TARGET attribute because its target is designated by PROTECTED_TARGET pointer 'p'
    w => p
    !ERROR: Pointer 'ws' must have the PROTECTED_TARGET attribute because its target is designated by PROTECTED_TARGET pointer 'ps'
    ws => ps(1:1)
    !ERROR: Pointer 'w' must have the PROTECTED_TARGET attribute because its target is designated by PROTECTED_TARGET pointer 'pt'
    w => pt%v
    !ERROR: Pointer 'w' must have the PROTECTED_TARGET attribute because its target is designated by PROTECTED_TARGET pointer 'pq'
    w => tt%pq
    !ERROR: Pointer 'w' must have the PROTECTED_TARGET attribute because its target is designated by PROTECTED_TARGET pointer 'ptresult'
    w => ptResult()
    w => plainResult()
    tt = t(1., w, p)
    tt = t(1., pt%q, pt%pq)
    !ERROR: Pointer 'q' must have the PROTECTED_TARGET attribute because its target is designated by PROTECTED_TARGET pointer 'p'
    tt = t(1., p, p)
    !ERROR: Pointer 'q' must have the PROTECTED_TARGET attribute because its target is designated by PROTECTED_TARGET pointer 'ps'
    tt = t(v=1., q=ps(1), pq=ps(2))
    print *, associated(p), associated(w, p), size(ps)
  end
  subroutine arguments(x)
    real, target :: x
    real, pointer :: w
    real, pointer, protected_target :: p, ps(:)
    type(t), pointer, protected_target :: pt
    type(ta), pointer, protected_target :: pa
    integer, pointer :: n
    external :: ext
    p => x
    call plainPointer(w)
    call ptPointer(w)
    call ptPointer(p)
    call ptOutPointer(p)
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute because its actual argument is designated by PROTECTED_TARGET pointer 'p'
    call plainPointer(p)
    call gen(w)
    call gen(n)
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute because its actual argument is designated by PROTECTED_TARGET pointer 'p'
    call gen(p)
    call ptInPointer(ps)
    call ptInPointer(ps(1:1))
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute because its actual argument is designated by PROTECTED_TARGET pointer 'ps'
    call inPointer(ps(1:1))
    call inIntent(p)
    call inIntent(ps(1))
    call inIntent(ptResult())
    call noIntent(pt%q)
    !ERROR: Nonpointer dummy argument 'a=' must have INTENT(IN) because its actual argument is designated by PROTECTED_TARGET pointer 'p'
    call noIntent(p)
    !ERROR: Nonpointer dummy argument 'a=' must have INTENT(IN) because its actual argument is designated by PROTECTED_TARGET pointer 'pt'
    call noIntent(pt%v)
    !ERROR: Nonpointer dummy argument 'a=' must have INTENT(IN) because its actual argument is designated by PROTECTED_TARGET pointer 'ptresult'
    call noIntent(ptResult())
    !ERROR: Nonpointer dummy argument 'a=' must have INTENT(IN) because its actual argument is designated by PROTECTED_TARGET pointer 'p'
    call valueArg(p)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'a=' is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
    call outIntent(p)
    call allocIn(pa%a)
    !ERROR: Nonpointer dummy argument 'a=' must have INTENT(IN) because its actual argument is designated by PROTECTED_TARGET pointer 'pa'
    call allocNoIntent(pa%a)
    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call allocNoIntent(ps)
    !ERROR: Nonpointer dummy argument 'a=' must have INTENT(IN) because its actual argument is designated by PROTECTED_TARGET pointer 'ps'
    call allocTkr(ps)
    call ext(pt%q)
    call ext(p + 1.)
    !ERROR: Actual argument designated by PROTECTED_TARGET pointer 'p' requires an explicit interface
    call ext(p)
    !ERROR: Actual argument designated by PROTECTED_TARGET pointer 'ps' requires an explicit interface
    call ext(ps(1))
  end
  ! PROTECTED_TARGET is not a characteristic (F2028 draft 15.3.2.2, 15.3.3),
  ! so it does not affect interface compatibility; C872 applies to the
  ! interface through which a procedure is referenced.
  subroutine characteristics(x)
    real, target :: x
    real, pointer, protected_target :: p
    procedure(ptPointer), pointer :: pp1
    procedure(plainPointer), pointer :: pp2
    procedure(plainResult), pointer :: fp1
    procedure(ptResult), pointer :: fp2
    p => x
    pp1 => plainPointer
    pp2 => ptPointer
    fp1 => ptResult
    fp2 => plainResult
    call pp1(p)
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute because its actual argument is designated by PROTECTED_TARGET pointer 'p'
    call pp2(p)
  end
end

submodule (m) sm
 contains
  module subroutine separate(a)
    real, pointer :: a
  end
end
