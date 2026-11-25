! Check SystemZ extension attributes on indirect call arguments.

! REQUIRES: systemz-registered-target

! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
! Only the -S run reaches the SystemZ ABI verifier, which also checks calls
! and returns with no CHECK line (assertions builds only).
! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -S %s -o /dev/null

! CHECK-LABEL: define void @s_(
! CHECK: call void %{{.*}}(i32 signext %{{.*}}, i32 signext %{{.*}}, i16 signext %{{.*}})
! CHECK: call {{.*}}i32 %{{.*}}(i32 signext %{{.*}})
subroutine s(p, f, k)
  interface
    subroutine sub(i, l, h)
      integer(4), value :: i
      logical(4), value :: l
      integer(2), value :: h
    end subroutine
    integer(4) function fn(i)
      integer(4), value :: i
    end function
  end interface
  procedure(sub), pointer :: p
  procedure(fn), pointer :: f
  integer(4) :: k, r
  logical(4) :: l
  integer(2) :: h
  call p(k, l, h)
  r = f(k)
end subroutine
