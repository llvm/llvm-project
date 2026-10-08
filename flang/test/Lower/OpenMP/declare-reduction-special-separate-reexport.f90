! Facade re-export of an operator-less declare reduction (a special function such
! as max/min, or an intrinsic operator on an intrinsic type). Such a reduction has
! a mangled symbol name that is not valid Fortran and, unlike a defined-operator
! reduction, has no re-exported operator to recover through. It is re-exported by
! a plain USE of the defining module, so it comes in as a single shared
! use-association rather than a facade-owned duplicate: reaching it through both
! the base and the facade must bind ONE reduction with the user's combiner (not a
! silent fallback to the intrinsic), and a re-exporting module must not re-emit a
! directive that would re-resolve in the wrong scope.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp sp_base.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp sp_wrap.f90
! The mangled reduction is re-exported by a plain USE, not an invalid item or a
! re-emitted directive (which would fork a facade-owned duplicate).
! RUN: FileCheck --check-prefix=MODFILE --input-file=sp_wrap.mod sp_wrap.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp sp_use.f90 -o - | FileCheck sp_use.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp sp_dual.f90 -o - | FileCheck sp_dual.f90

!--- sp_base.f90
! The combiner is deliberately not the intrinsic max, so a silent fallback to the
! intrinsic reduction would be observable.
module sp_base
  !$omp declare reduction(max:integer:omp_out=omp_out*omp_in) &
  !$omp   initializer(omp_priv=1)
end module

!--- sp_wrap.f90
! MODFILE: use sp_base
! MODFILE-NOT: only:op.max
! MODFILE-NOT: DECLARE REDUCTION
module sp_wrap
  use sp_base
end module

!--- sp_use.f90
! The user's product combiner is preserved (muli), and the op is owned by the
! defining module, not the facade.
! CHECK: omp.declare_reduction @[[RED:_QQMsp_baseop\.max_i32]] : i32
! CHECK: arith.muli
! CHECK: omp.wsloop
! CHECK-SAME: reduction(@[[RED]]
! CHECK-NOT: not yet implemented
program main
  use sp_wrap
  integer :: x, i
  x = 1
  !$omp parallel do reduction(max:x)
  do i = 1, 5
    x = x * i
  end do
  print *, x
end program

!--- sp_dual.f90
! Reaching the reduction through both the base and the facade binds ONE reduction
! with the user combiner (muli), not the intrinsic (maxsi).
! CHECK: omp.declare_reduction @[[RED:_QQMsp_baseop\.max_i32]] : i32
! CHECK: arith.muli
! CHECK-NOT: arith.maxsi
! CHECK: omp.wsloop
! CHECK-SAME: reduction(@[[RED]]
program main
  use sp_base
  use sp_wrap
  integer :: x, i
  x = 1
  !$omp parallel do reduction(max:x)
  do i = 1, 5
    x = x * i
  end do
  print *, x
end program

! A facade that makes the reduction PRIVATE must not re-export it: a consumer must
! fall back to the intrinsic reduction, exactly as without a module-file
! round-trip (not silently bind the private base reduction).
! RUN: %flang_fc1 -fsyntax-only -fopenmp pr_base.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp pr_facade.f90
! RUN: FileCheck --check-prefix=PRIVMOD --input-file=pr_facade.mod pr_facade.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp pr_use.f90 -o - | FileCheck pr_use.f90

!--- pr_base.f90
module pr_base
  !$omp declare reduction(max:integer:omp_out=omp_out*omp_in) &
  !$omp   initializer(omp_priv=1)
end module

!--- pr_facade.f90
! The private reduction is not re-exported: no bare `use pr_base`.
! PRIVMOD-NOT: use pr_base
! PRIVMOD-NOT: DECLARE REDUCTION
module pr_facade
  use pr_base
  private
end module

!--- pr_use.f90
! The consumer binds the intrinsic max reduction, not the private base one.
! CHECK: omp.declare_reduction @[[RED:max_i32]] : i32
! CHECK: arith.maxsi
! CHECK-NOT: arith.muli
! CHECK-NOT: op.max
! CHECK: omp.wsloop
! CHECK-SAME: reduction(@[[RED]]
program main
  use pr_facade
  integer :: x, i
  x = 1
  !$omp parallel do reduction(max:x)
  do i = 1, 5
    x = max(x, i)
  end do
  print *, x
end program

! Two facades of the SAME base compiled in ONE invocation must each re-export the
! reduction (the per-module re-export bookkeeping must reset between module
! files); otherwise the second facade silently drops it and its consumers fall
! back to the intrinsic.
! RUN: %flang_fc1 -fsyntax-only -fopenmp two.f90
! RUN: FileCheck --check-prefix=FA --input-file=xfa.mod two.f90
! RUN: FileCheck --check-prefix=FB --input-file=xfb.mod two.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp xfa_use.f90 -o - | FileCheck --check-prefix=UA xfa_use.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp xfb_use.f90 -o - | FileCheck --check-prefix=UB xfb_use.f90

!--- two.f90
! FA: use xb
! FB: use xb
module xb
  !$omp declare reduction(max:integer:omp_out=omp_out*omp_in) &
  !$omp   initializer(omp_priv=1)
end module
module xfa
  use xb
end module
module xfb
  use xb
end module

!--- xfa_use.f90
! UA: omp.wsloop
! UA-SAME: reduction(@_QQMxbop.max_i32
program main
  use xfa
  integer :: x, i
  x = 1
  !$omp parallel do reduction(max:x)
  do i = 1, 5
    x = x * i
  end do
  print *, x
end program

!--- xfb_use.f90
! The second facade must bind the SAME user reduction, not the intrinsic.
! UB: omp.wsloop
! UB-SAME: reduction(@_QQMxbop.max_i32
program main
  use xfb
  integer :: x, i
  x = 1
  !$omp parallel do reduction(max:x)
  do i = 1, 5
    x = x * i
  end do
  print *, x
end program
