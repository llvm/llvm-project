! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s
! Continuation before C style comment.

integer :: i

! Single line comment.
! CHECK: i=1
/* comment 1 */
i&
/* comment 2 */
=1
! CHECK: i=2
  /* comment 1 */
  i&
  /* comment 2 */
  =2
! CHECK: i=3
  /* comment 1 */
  i&  
  /* comment 2 */  
  =3
! CHECK: i=4
  /* comment 1 */
  i&  /* inline comment */  
  /* comment 2 */  
  =4
! CHECK: i=42
i&
/* c */ &= 42
! CHECK: i=44
i = 43 &
/* c */ & + 1

! Multi-line comment.
! CHECK: i=5
i&
/* c
*/ = 5

! Compiler directive.
! CHECK: !$OMP PARALLEL
! CHECK: !$OMP END PARALLEL
!$omp para&
/* comment */
!$omp llel
!$omp end parallel
! CHECK: !$OMP PARALLEL
! CHECK: !$OMP END PARALLEL
!$omp para&
/* multi
 * line */
!$omp llel
!$omp end parallel
! CHECK: !$OMP PARALLEL DO
! CHECK: !$OMP END PARALLEL DO
!$omp parallel do &
/* c */ !$omp private(i)
  do i = 1, 10
  end do
!$omp end parallel do

! Source line continuation after macro expansion.
! CHECK: i=12
! CHECK: i=14
! CHECK: PRINT *, "pass"
#define CONT &
i = 6 CONT
/* comment */
+ 6
i = 7 CONT
/* multi
 * line */ + 7
print *,'pass'
end
