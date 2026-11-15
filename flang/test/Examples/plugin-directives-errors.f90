! Check the semantic errors in the directives of a plugin loaded with
! `flang -fc1 -load`.

! REQUIRES: plugins, examples
! XFAIL: system-aix

! RUN: rm -rf %t && mkdir -p %t
! RUN: not %flang_fc1 -load %llvmshlibdir/flangDirectivesPlugin%pluginext \
! RUN:   -fsyntax-only -module-dir %t %s 2>&1 | FileCheck %s

module m_errors
  real :: x
  type t
  end type
  interface gen
    module procedure gen1, gen2
  end interface
  ! CHECK: error: A 'example watch' directive must name what it applies to, or appear in a subprogram
  !dir$ example watch(by=x)
  ! CHECK: error: Unknown 'example' directive 'nothing'
  !dir$ example nothing
contains
  subroutine gen1(a)
    integer :: a
  end
  subroutine gen2(a)
    real :: a
  end
  subroutine s(dummy, pp)
    external dummy
    procedure(), pointer :: pp
    integer :: sf, i
    sf(i) = i
    ! CHECK: error: The 'example callback' directive requires argument 'handler'
    !dir$ example callback(priority=1)
    ! CHECK: error: 'priority' is not an argument of the 'example watch' directive
    !dir$ example watch(x, priority=1)
    ! CHECK: error: Argument 'priority' must be an integer
    !dir$ example callback(handler=s, priority=high)
    ! CHECK: error: Argument 'priority' must be an integer
    !dir$ example callback(handler=s, priority=1.5)
    ! CHECK: error: Argument 'handler' appears more than once
    !dir$ example callback(handler=s, handler=s)
    ! CHECK: error: Only the first argument of a 'example callback' directive may be positional
    !dir$ example callback(handler=s, s)
    ! CHECK: error: 'x' is not a procedure
    !dir$ example callback(handler=x)
    ! CHECK: error: 's' is not a variable
    !dir$ example watch(s)
    ! CHECK: error: 't' is not a variable
    !dir$ example watch(x, by=t)
    ! CHECK: error: 'undeclared' is not declared
    !dir$ example callback(handler=undeclared)
    ! CHECK: error: COMMON block /nocommon/ is not declared
    !dir$ example watch(/nocommon/)
    ! CHECK: error: 'gen' is a generic interface; argument 'handler' of a 'example callback' directive must name a specific procedure
    !dir$ example callback(handler=gen)
    ! CHECK: error: 'gen' is a generic interface; a 'example callback' directive must name one of its specific procedures
    !dir$ example callback(gen, handler=s)
    ! CHECK: error: 'dummy' must be a subprogram or an external procedure
    !dir$ example callback(handler=dummy)
    ! CHECK: error: 'pp' must be a subprogram or an external procedure
    !dir$ example callback(handler=pp)
    ! CHECK: error: 'sf' must be a subprogram or an external procedure
    !dir$ example callback(handler=sf)
  end
end
