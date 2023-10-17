! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for distinguishability of defined I/O procedures defined within
! and outside their types.
module m1
  type t1
    integer n
   contains
    procedure :: readt1a, readt1b
    !ERROR: Generic 'read(unformatted)' may not have specific procedures 'readt1a' and 'readt1b' as their interfaces are not distinguishable
    generic :: read(unformatted) => readt1a, readt1b
  end type
  type t2
    integer n
  end type
  type t3
    integer n
  end type
  !ERROR: Generic 'read(unformatted)' may not have specific procedures 'readt2a' and 'readt2b' as their interfaces are not distinguishable
  interface read(unformatted)
    module procedure :: readt1a, readt2a, readt2b, readt3
  end interface
 contains
#define DEFINE_READU(name, type) \
  subroutine name(dtv, unit, iostat, iomsg); \
    class(type), intent(in out) :: dtv; \
    integer, intent(in) :: unit; \
    integer, intent(out) :: iostat; \
    character(*), intent(in out) :: iomsg; \
    read(unit, iostat=iostat, iomsg=iomsg) dtv%n; \
  end subroutine name
  !ERROR: Derived type 't1' has conflicting type-bound input/output procedure 'read(unformatted)'
  DEFINE_READU(readt1a, t1)
  DEFINE_READU(readt1b, t1)
  DEFINE_READU(readt2a, t2)
  DEFINE_READU(readt2b, t2)
  DEFINE_READU(readt3, t3)
end module
