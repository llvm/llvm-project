!RUN: %flang_fc1 -fsyntax-only -fhermetic-module-files -DSTEP=1 %s
!RUN: %flang_fc1 -fsyntax-only -DSTEP=2 %s
!RUN: not %flang_fc1 -fsyntax-only -pedantic %s 2>&1 | FileCheck %s

! Tests that a module captured in a hermetic module file is compatible when
! USE'd with a module of the same name USE'd directly.

#if STEP == 1
module modfile71a
  ! not errors
  integer, parameter :: good_named_const = 123
  integer :: good_var = 1
  type :: good_derived
    integer component
  end type
  procedure(), pointer :: good_proc_ptr
  generic :: gen => bad_subroutine
  ! bad, but okay if unused
  integer, parameter :: unused_bad_named_const = 123
  integer :: unused_bad_var = 1
  type :: unused_bad_derived
    integer component
  end type
  procedure(), pointer :: unused_bad_proc_ptr
  ! errors
  integer, parameter :: bad_named_const = 123
  integer :: bad_var = 1
  type :: bad_derived
    integer component
  end type
  procedure(), pointer :: bad_proc_ptr
 contains
  subroutine good_subroutine
  end
  subroutine unused_bad_subroutine(x)
    integer x
  end
  subroutine bad_subroutine(x)
    integer x
  end
end

module modfile71b
  use modfile71a ! capture hermetically
end

#elif STEP == 2
module modfile71a
  ! not errors
  integer, parameter :: good_named_const = 123
  integer :: good_var = 1
  type :: good_derived
    integer component
  end type
  procedure(), pointer :: good_proc_ptr
  generic :: gen => bad_subroutine
  ! bad, but okay if unused
  integer, parameter :: unused_bad_named_const = 666
  real :: unused_bad_var = 1.
  type :: unused_bad_derived
    real component
  end type
  real, pointer :: unused_bad_proc_ptr
  ! errors
  integer, parameter :: bad_named_const = 666
  real :: bad_var = 1.
  type :: bad_derived
    real component
  end type
  real, pointer :: bad_proc_ptr
 contains
  subroutine good_subroutine
  end
  subroutine unused_bad_subroutine(x)
    real x
  end
  subroutine bad_subroutine(x)
    real x
  end
end

#else

!CHECK: warning: 'bad_derived' is use-associated from 'bad_derived' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'bad_named_const' is use-associated from 'bad_named_const' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'bad_proc_ptr' is use-associated from 'bad_proc_ptr' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'bad_subroutine' is use-associated from 'bad_subroutine' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'bad_var' is use-associated from 'bad_var' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'good_derived' is use-associated from 'good_derived' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'good_named_const' is use-associated from 'good_named_const' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'good_proc_ptr' is use-associated from 'good_proc_ptr' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'good_subroutine' is use-associated from 'good_subroutine' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'good_var' is use-associated from 'good_var' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'unused_bad_derived' is use-associated from 'unused_bad_derived' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'unused_bad_named_const' is use-associated from 'unused_bad_named_const' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'unused_bad_proc_ptr' is use-associated from 'unused_bad_proc_ptr' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'unused_bad_subroutine' is use-associated from 'unused_bad_subroutine' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'unused_bad_var' is use-associated from 'unused_bad_var' in two distinct instances of module 'modfile71a'
!CHECK: error: Reference to 'bad_derived' is ambiguous
!CHECK: error: Reference to 'bad_named_const' is ambiguous
!CHECK: error: Reference to 'bad_var' is ambiguous
!CHECK: error: Reference to 'bad_proc_ptr' is ambiguous
!CHECK: error: Reference to 'bad_subroutine' is ambiguous
!CHECK-NOT: error:
!CHECK-NOT: warning:

program main
  use modfile71a
  use modfile71b
  type(good_derived) goodx
  type(bad_derived) badx
  print *, good_named_const
  good_var = 1
  good_proc_ptr => null()
  call good_subroutine
  print *, bad_named_const
  print *, bad_var
  bad_proc_ptr => null()
  call bad_subroutine(1)
end
#endif
