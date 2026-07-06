! Facade re-export of an operator-less declare reduction (a special function such
! as max/min, or an intrinsic operator on an intrinsic type). Such a reduction has
! a mangled symbol name that is not valid Fortran and, unlike a defined-operator
! reduction, has no re-exported operator to recover through. It is re-exported by
! a plain USE of the defining module, so a facade's module file carries a shared
! `use` of the defining module rather than an invalid use-only item or a re-emitted
! directive. This test covers the module-file writing. The lowering of the imported
! reduction is added with the multi-type user-defined reduction lowering support.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp sp_base.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp sp_wrap.f90
! The mangled reduction is re-exported by a plain USE, not an invalid item or a
! re-emitted directive (which would fork a facade-owned duplicate).
! RUN: FileCheck --check-prefix=MODFILE --input-file=sp_wrap.mod sp_wrap.f90

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

! A facade that makes the reduction PRIVATE must not re-export it: its module file
! must not carry a bare `use pr_base` (a consumer then falls back to the intrinsic
! reduction, not the private base one).
! RUN: %flang_fc1 -fsyntax-only -fopenmp pr_base.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp pr_facade.f90
! RUN: FileCheck --check-prefix=PRIVMOD --input-file=pr_facade.mod pr_facade.f90

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

! Two facades of the SAME base compiled in ONE invocation must each re-export the
! reduction in their module file (the per-module re-export bookkeeping must reset
! between module files); otherwise the second facade silently drops it.
! RUN: %flang_fc1 -fsyntax-only -fopenmp two.f90
! RUN: FileCheck --check-prefix=FA --input-file=xfa.mod two.f90
! RUN: FileCheck --check-prefix=FB --input-file=xfb.mod two.f90

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
