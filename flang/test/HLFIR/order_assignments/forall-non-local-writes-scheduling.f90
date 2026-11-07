! Test forall scheduling analysis from lower-hlfir-ordered-assignments pass
! when evaluations inside FORALL write to non local variables. This happens
! with the STAT= argument of NEXT/PREVIOUS, and with PURE functions that have
! INTENT(OUT) or INTENT(INOUT) dummy arguments (an extension).
! The printed output is done via LLVM_DEBUG, hence the "asserts" requirement.

! RUN: bbc -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments{fuse-assignments=false})" --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOFUSE
! RUN: bbc -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments{fuse-assignments=true})" --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s --check-prefixes=CHECK,FUSE
! REQUIRES: asserts

module forall_writes_m
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
  interface
    pure integer function setstat(x, s)
      integer, intent(in) :: x
      integer, intent(inout) :: s
    end function
    elemental integer function esetstat(x, s)
      integer, intent(in) :: x
      integer, intent(inout) :: s
    end function
    pure integer function readonly(x, s)
      integer, intent(in) :: x
      integer, intent(in) :: s
    end function
  end interface
  type tb
  contains
    procedure, nopass :: tbp => setstat
  end type
  type tcp
    procedure(cf), nopass, pointer :: p
  end type
contains
  pure character(2) function cf()
    cf = "01"
  end function
end module

! The STAT= write must be visible to the second assignment: no fusion.
subroutine next_stat_then_read(a, n, st, r)
  use forall_writes_m
  type(color) :: a(3), n(3)
  integer :: st(3), r(3), i
  forall (i = 1:3)
    n(i) = next(a(i), stat=st(i))
    r(i) = st(4-i)
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPnext_stat_then_read ------------
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 1 save  (w): forall/region_assign1/rhs
!CHECK-NEXT: run 2 evaluate: forall/region_assign1
!NOFUSE-NEXT: run 3 evaluate: forall/region_assign2
!FUSE-NEXT: run 2 evaluate: forall/region_assign2

! The read of the first assignment must happen before the STAT= write.
subroutine read_then_next_stat(a, n, st, t)
  use forall_writes_m
  type(color) :: a(3), n(3)
  integer :: st(3), t(3), i
  forall (i = 1:3)
    t(i) = st(4-i)
    n(i) = next(a(i), stat=st(i))
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPread_then_next_stat ------------
!CHECK-NEXT: run 1 evaluate: forall/region_assign1
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 2 save  (w): forall/region_assign2/rhs
!CHECK-NEXT: run 3 evaluate: forall/region_assign2

! The mask must be saved before the STAT= write.
subroutine mask_and_next_stat(a, n, st, r)
  use forall_writes_m
  type(color) :: a(3), n(3)
  integer :: st(3), r(3), i
  forall (i = 1:3, st(i) == 0)
    n(i) = next(a(i), stat=st(i))
    r(i) = 1
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPmask_and_next_stat ------------
!CHECK-NEXT: conflict: R/W: <block argument> {{.*}} at index: 2 W:<block argument> {{.*}} at index: 2
!CHECK-NEXT: run 1 save    : forall/forall_mask1/mask
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 2 save  (w): forall/forall_mask1/region_assign1/rhs
!CHECK-NEXT: run 3 evaluate: forall/forall_mask1/region_assign1
!NOFUSE-NEXT: run 4 evaluate: forall/forall_mask1/region_assign2
!FUSE-NEXT: run 3 evaluate: forall/forall_mask1/region_assign2

! A mask with a STAT= write is evaluated only once.
subroutine next_stat_in_mask(a, st, r, c)
  use forall_writes_m
  type(color) :: a(3)
  integer :: st(3), r(3), c(3), i
  forall (i = 1:3, next(a(i), stat=st(i)) /= blue)
    r(i) = st(i)
    c(i) = st(4-i)
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPnext_stat_in_mask ------------
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 1 save  (w): forall/forall_mask1/mask
!CHECK-NEXT: run 2 evaluate: forall/forall_mask1/region_assign1
!NOFUSE-NEXT: run 3 evaluate: forall/forall_mask1/region_assign2
!FUSE-NEXT: run 2 evaluate: forall/forall_mask1/region_assign2

! The STAT= write after a call in the same region is still found.
subroutine call_then_next_stat_in_mask(a, x, st, r)
  use forall_writes_m
  type(color) :: a(3)
  integer :: x(3), st(3), r(3), i
  forall (i = 1:3, readonly(x(i), x(i)) > 0 .and. next(a(i), stat=st(i)) /= blue)
    r(i) = st(4-i)
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPcall_then_next_stat_in_mask ------------
!CHECK-NEXT: unknown effect: {{.*}} = fir.call @_QPreadonly(
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 1 save  (w): forall/forall_mask1/mask
!CHECK-NEXT: run 2 evaluate: forall/forall_mask1/region_assign1

! The STAT= write to a whole OPTIONAL argument is nested in a presence test
! (fir.if) and is still found.
subroutine optional_next_stat_in_mask(a, s, r)
  use forall_writes_m
  type(color) :: a(3)
  integer, optional :: s
  integer :: r(3), i
  forall (i = 1:3, next(a(i), stat=s) /= blue)
    r(i) = s
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPoptional_next_stat_in_mask ------------
!CHECK-NOT: run
!CHECK: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 1 save  (w): forall/forall_mask1/mask
!CHECK-NEXT: run 2 evaluate: forall/forall_mask1/region_assign1

! The STAT= write after an operation without memory effect interface (the
! llvm.intr.stacksave for the character function result) is still found.
subroutine stacksave_then_next_stat_in_mask(x, a, st, r)
  use forall_writes_m
  type(tcp) :: x(3)
  type(color) :: a(3)
  integer :: st(3), r(3), i
  forall (i = 1:3, x(i)%p() == "01" .and. next(a(i), stat=st(i)) /= blue)
    r(i) = st(4-i)
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPstacksave_then_next_stat_in_mask ------------
!CHECK-NEXT: unknown effect: {{.*}} = llvm.intr.stacksave
!CHECK-NEXT: unknown effect: {{.*}} = fir.call
!CHECK-NEXT: unknown effect: llvm.intr.stackrestore
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 1 save  (w): forall/forall_mask1/mask
!CHECK-NEXT: run 2 evaluate: forall/forall_mask1/region_assign1

! The mask must be saved before the write by the relaxed PURE function.
subroutine mask_and_relaxed_pure(a, n, st, r)
  use forall_writes_m
  integer :: a(3), n(3), st(3), r(3), i
  forall (i = 1:3, st(i) == 0)
    n(i) = setstat(a(i), st(i))
    r(i) = 1
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPmask_and_relaxed_pure ------------
!CHECK-NEXT: unknown effect: {{.*}} = fir.call @_QPsetstat({{.*}} {fir.may_be_modified}
!CHECK-NEXT: conflict: R/W: <block argument> {{.*}} at index: 2 W:<block argument> {{.*}} at index: 2
!CHECK-NEXT: run 1 save    : forall/forall_mask1/mask
!CHECK-NEXT: unknown effect: {{.*}} = fir.call @_QPsetstat(
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 2 save  (w): forall/forall_mask1/region_assign1/rhs
!CHECK-NEXT: run 3 evaluate: forall/forall_mask1/region_assign1
!NOFUSE-NEXT: run 4 evaluate: forall/forall_mask1/region_assign2
!FUSE-NEXT: run 3 evaluate: forall/forall_mask1/region_assign2

! Same through a procedure pointer.
subroutine mask_and_relaxed_pure_pointer(a, n, st, r)
  use forall_writes_m
  procedure(setstat), pointer :: p
  integer :: a(3), n(3), st(3), r(3), i
  p => setstat
  forall (i = 1:3, st(i) == 0)
    n(i) = p(a(i), st(i))
    r(i) = 1
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPmask_and_relaxed_pure_pointer ------------
!CHECK-NEXT: unknown effect: {{.*}} = fir.call %{{.*}}({{.*}} {fir.may_be_modified}
!CHECK-NEXT: conflict: R/W: <block argument> {{.*}} at index: 2 W:<block argument> {{.*}} at index: 2
!CHECK-NEXT: run 1 save    : forall/forall_mask1/mask
!CHECK-NEXT: unknown effect: {{.*}} = fir.call %{{.*}}(
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 2 save  (w): forall/forall_mask1/region_assign1/rhs
!CHECK-NEXT: run 3 evaluate: forall/forall_mask1/region_assign1
!NOFUSE-NEXT: run 4 evaluate: forall/forall_mask1/region_assign2
!FUSE-NEXT: run 3 evaluate: forall/forall_mask1/region_assign2

! Same through a type-bound procedure.
subroutine mask_and_relaxed_pure_dispatch(o, a, n, st, r)
  use forall_writes_m
  class(tb) :: o
  integer :: a(3), n(3), st(3), r(3), i
  forall (i = 1:3, st(i) == 0)
    n(i) = o%tbp(a(i), st(i))
    r(i) = 1
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPmask_and_relaxed_pure_dispatch ------------
!CHECK-NEXT: unknown effect: {{.*}} = fir.dispatch "tbp"({{.*}} arg_attrs([{}, {fir.may_be_modified}])
!CHECK-NEXT: conflict: R/W: <block argument> {{.*}} at index: 3 W:<block argument> {{.*}} at index: 3
!CHECK-NEXT: run 1 save    : forall/forall_mask1/mask
!CHECK-NEXT: unknown effect: {{.*}} = fir.dispatch "tbp"(
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 2 save  (w): forall/forall_mask1/region_assign1/rhs
!CHECK-NEXT: run 3 evaluate: forall/forall_mask1/region_assign1
!NOFUSE-NEXT: run 4 evaluate: forall/forall_mask1/region_assign2
!FUSE-NEXT: run 3 evaluate: forall/forall_mask1/region_assign2

! A WHERE mask inside FORALL must be saved before the write.
subroutine where_mask_and_relaxed_pure(a, n, st, r)
  use forall_writes_m
  integer :: a(2,3), n(2,3), st(2,3), r(2,3), i
  forall (i = 1:2)
    where (st(i,:) == 0)
      n(i,:) = esetstat(a(i,:), st(i,:))
      r(i,:) = 1
    end where
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPwhere_mask_and_relaxed_pure ------------
!CHECK-NEXT: unknown effect: {{.*}} = fir.call @_QPesetstat({{.*}} {fir.may_be_modified}
!CHECK-NEXT: conflict: R/W: <block argument> {{.*}} at index: 2 W:<block argument> {{.*}} at index: 2
!CHECK-NEXT: run 1 save    : forall/where1/mask
!CHECK-NEXT: unknown effect: {{.*}} = fir.call @_QPesetstat(
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 2 save  (w): forall/where1/region_assign1/rhs
!CHECK-NEXT: run 3 evaluate: forall/where1/region_assign1
!NOFUSE-NEXT: run 4 evaluate: forall/where1/region_assign2
!FUSE-NEXT: run 3 evaluate: forall/where1/region_assign2

! A nested FORALL bound must be saved before the write.
subroutine nested_bound_and_relaxed_pure(a, n, r, k)
  use forall_writes_m
  integer :: a(2,3), n(2,200), r(2,200), i, j, k
  forall (i = 1:2)
    forall (j = 1:k)
      n(i,j) = setstat(a(i,j), k)
      r(i,j) = 1
    end forall
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPnested_bound_and_relaxed_pure ------------
!CHECK-NEXT: unknown effect: {{.*}} = fir.call @_QPsetstat({{.*}} {fir.may_be_modified}
!CHECK-NEXT: conflict: R/W: <block argument> {{.*}} at index: 3 W:<block argument> {{.*}} at index: 3
!CHECK-NEXT: run 1 save    : forall/forall1/ub
!CHECK-NEXT: unknown effect: {{.*}} = fir.call @_QPsetstat(
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 2 save  (w): forall/forall1/region_assign1/rhs
!CHECK-NEXT: run 3 evaluate: forall/forall1/region_assign1
!NOFUSE-NEXT: run 4 evaluate: forall/forall1/region_assign2
!FUSE-NEXT: run 3 evaluate: forall/forall1/region_assign2

! PURE functions with INTENT(IN) arguments do not write: the mask is not saved
! (the RHS is saved because the call may read the assigned variable).
subroutine mask_and_pure(a, n, st, r)
  use forall_writes_m
  integer :: a(3), n(3), st(3), r(3), i
  forall (i = 1:3, st(i) == 0)
    n(i) = readonly(a(i), st(i))
    r(i) = 1
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPmask_and_pure ------------
!CHECK-NEXT: unknown effect: {{.*}} = fir.call @_QPreadonly({{.*}}) {{.*}} : (!fir.ref<i32>, !fir.ref<i32>) -> i32
!CHECK-NEXT: unknown effect: {{.*}} = fir.call @_QPreadonly
!CHECK-NEXT: conflict: R/W: <unknown> W:<block argument> {{.*}} at index: 1
!CHECK-NEXT: run 1 save    : forall/forall_mask1/region_assign1/rhs
!CHECK-NEXT: run 2 evaluate: forall/forall_mask1/region_assign1
!NOFUSE-NEXT: run 3 evaluate: forall/forall_mask1/region_assign2
!FUSE-NEXT: run 2 evaluate: forall/forall_mask1/region_assign2

! Writes to unknown locations in pure evaluations (here the derived type
! assignment to the structure constructor temporary) are ignored: the mask is
! not saved.
subroutine mask_and_structure_constructor(a, b, n)
  type t2
    integer, allocatable :: c
  end type
  integer :: n, b(n), i
  type(t2) :: a(n)
  forall (i = 1:n, b(i) > 0) a(i) = t2(b(i))
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPmask_and_structure_constructor ------------
!CHECK-NOT: forall_mask1/mask
!CHECK: conflict: R/W: <unknown> W:<block argument> {{.*}} at index: 0
!CHECK-NEXT: run 1 save    : forall/forall_mask1/region_assign1/rhs
!CHECK-NEXT: run 2 evaluate: forall/forall_mask1/region_assign1
