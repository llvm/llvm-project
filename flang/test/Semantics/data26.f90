! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! F'2023 19.4 p5: a data-implied-do index variable takes the type of its
! name in the scoping unit -- including its kind, and even when the
! declaration follows the DATA statement.
! CHECK: Subprogram scope: s
! CHECK: i size=8 offset={{[0-9]+}}: ObjectEntity type: INTEGER(8)
! CHECK: ImpliedDos scope:
! CHECK: i size=8 offset=0: ObjectEntity type: INTEGER(8)
subroutine s
  logical, dimension(4), save :: util
  data (util(i),i=1,4)/4*.true./
  integer(8) :: i
end subroutine
