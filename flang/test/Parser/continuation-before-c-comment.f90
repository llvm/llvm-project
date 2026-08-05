! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s
! Continuation before C style comment.

integer :: i

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

! CHECK: !$OMP PARALLEL DO
  /* comments before directives are allowed now */ !$omp parallel do
  do i = 1, 10
  end do

! CHECK: PRINT *, "pass"
  /* C comment */
  print *,'pass'
end
