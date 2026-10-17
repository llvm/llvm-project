! The unsupported cases cover the following matrix. "element UDR" expects
! the more specific user-defined reduction diagnostic; all other cases expect
! the complex-part diagnostic.
!
! case  part  object   reduction  inner construct                    map
!   0   real  scalar   intrinsic  parallel do                       implicit
!   1   imag  scalar   UDR        parallel do                       explicit
!   2   real  element  UDR        parallel do                       implicit
!   3   imag  element  intrinsic  parallel do                       explicit
!   4   imag  scalar   intrinsic  teams distribute parallel do      implicit
!   5   real  element  UDR        teams distribute parallel do      explicit
!   6   real  element  intrinsic  teams distribute parallel do      explicit
!   7   imag  element  UDR        teams distribute parallel do      implicit
!
! RUN: split-file %s %t
! RUN: %flang_fc1 -E -P -cpp -fopenmp -DPART=re %t/unsupported.F90 -o %t/case0.f90
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/case0.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/case0.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %flang_fc1 -E -P -cpp -fopenmp -DPART=im -DUDR -DEXPLICIT_MAP %t/unsupported.F90 -o %t/case1.f90
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/case1.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/case1.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %flang_fc1 -E -P -cpp -fopenmp -DPART=re -DARRAY_ELEMENT -DUDR %t/unsupported.F90 -o %t/case2.f90
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/case2.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/case2.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %flang_fc1 -E -P -cpp -fopenmp -DPART=im -DARRAY_ELEMENT -DEXPLICIT_MAP %t/unsupported.F90 -o %t/case3.f90
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/case3.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/case3.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %flang_fc1 -E -P -cpp -fopenmp -DPART=im -DTEAMS %t/unsupported.F90 -o %t/case4.f90
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/case4.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/case4.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %flang_fc1 -E -P -cpp -fopenmp -DPART=re -DARRAY_ELEMENT -DUDR -DTEAMS -DEXPLICIT_MAP %t/unsupported.F90 -o %t/case5.f90
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/case5.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/case5.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %flang_fc1 -E -P -cpp -fopenmp -DPART=re -DARRAY_ELEMENT -DTEAMS -DEXPLICIT_MAP %t/unsupported.F90 -o %t/case6.f90
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/case6.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/case6.f90 2>&1 | FileCheck %s --check-prefix=COMPLEX
! RUN: %flang_fc1 -E -P -cpp -fopenmp -DPART=im -DARRAY_ELEMENT -DUDR -DTEAMS %t/unsupported.F90 -o %t/case7.f90
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/case7.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/case7.f90 2>&1 | FileCheck %s --check-prefix=ELEMENT
! RUN: bbc -emit-hlfir -fopenmp -o - %t/supported.f90 | FileCheck %s --check-prefix=SUPPORTED
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %t/supported.f90 | FileCheck %s --check-prefix=SUPPORTED

! Diagnose inner reductions before implicit target mapping tries to lower
! complex parts. Preserve the more specific UDR array-element diagnostic.
! COMPLEX: not yet implemented: REDUCTION of a complex part
! ELEMENT: not yet implemented: REDUCTION of an array element using a user-defined reduction

!--- unsupported.F90
#ifdef ARRAY_ELEMENT
#define ITEM z(2)
#else
#define ITEM z
#endif
#ifdef TEAMS
#define INNER teams distribute parallel do
#else
#define INNER parallel do
#endif
#ifdef EXPLICIT_MAP
#define MAP map(tofrom:z)
#else
#define MAP
#endif
subroutine complex_part(z)
#ifdef ARRAY_ELEMENT
  complex :: z(4)
#else
  complex :: z
#endif
  integer :: i
#ifdef UDR
  !$omp declare reduction(+: real: omp_out=omp_out+omp_in) initializer(omp_priv=0)
#endif
  !$omp target INNER MAP reduction(+: ITEM%PART)
  do i = 1, 4
    ITEM%PART = ITEM%PART + real(i)
  end do
  !$omp end target INNER
end subroutine

!--- supported.f90
! SUPPORTED-LABEL: func.func @_QPwhole_complex
! SUPPORTED: omp.target
! SUPPORTED: omp.parallel
! SUPPORTED: omp.wsloop {{.*}}reduction(@add_reduction_z32 {{.*}} -> %[[RED:arg[0-9]+]]
! SUPPORTED: %[[DECL:.*]]:2 = hlfir.declare %[[RED]]
! SUPPORTED: fir.load %[[DECL]]#0
subroutine whole_complex(z)
  complex :: z
  integer :: i
  !$omp target parallel do reduction(+: z)
  do i = 1, 4
    z = z + cmplx(i,i)
  end do
  !$omp end target parallel do
end subroutine
