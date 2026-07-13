! RUN: %python %S/test_errors.py %s %flang_fc1
recursive subroutine sub(dp, dpp)
  procedure(inner) dp
  procedure(inner), pointer :: dpp
  procedure(inner) ext
  procedure(sub), pointer :: p1 => sub ! ok
  procedure(inner), pointer :: p2 => ext ! ok
  !ERROR: Procedure pointer 'p3' initializer 'inner' is neither an external nor a module procedure
  procedure(inner), pointer :: p3 => inner
  !ERROR: Procedure pointer 'p4' initializer 'dp' is neither an external nor a module procedure
  procedure(inner), pointer :: p4 => dp
  !ERROR: Procedure pointer 'p5' initializer 'dpp' is neither an external nor a module procedure
  procedure(inner), pointer :: p5 => dpp
  generic :: generic => ext
  !ERROR: 'generic' must be an abstract interface or a procedure with an explicit interface
  procedure(generic), pointer :: p6 ! => generic
 contains
  subroutine inner
  end
end
recursive function fun() result(res)
  procedure(fun), pointer :: p1 => fun ! ok
  !ERROR: Procedure pointer 'p2' initializer 'inner' is neither an external nor a module procedure
  procedure(inner), pointer :: p2 => inner
  res = 0.
 contains
  function inner()
    inner = 0.
  end
end
module m
  procedure(msub), pointer :: ps1 => msub ! ok
  procedure(mfun), pointer :: pf1 => mfun ! ok
 contains
  recursive subroutine msub
    procedure(msub), pointer :: ps2 => msub ! ok
    !ERROR: Procedure pointer 'ps3' initializer 'inner' is neither an external nor a module procedure
    procedure(inner), pointer :: ps3 => inner
   contains
    subroutine inner
    end
  end
  recursive function mfun() result(res)
    procedure(mfun), pointer :: pf2 => mfun ! ok
    !ERROR: Procedure pointer 'pf3' initializer 'inner' is neither an external nor a module procedure
    procedure(inner), pointer :: pf3 => inner
    res = 0.
   contains
    function inner()
      inner = 0.
    end
  end
end
