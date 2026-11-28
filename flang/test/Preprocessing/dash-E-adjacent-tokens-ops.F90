! RUN: %flang -E %s 2>&1 | FileCheck --strict-whitespace %s
! RUN: %flang -E %s > %t.f90
! RUN: %flang_fc1 -fdebug-unparse %t.f90 2>&1 | FileCheck --strict-whitespace --check-prefix=UNPARSE %s
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck --strict-whitespace --check-prefix=UNPARSE %s
! Tokens formed by macro expansion next to other characters, with no blank in
! between in the original source, are not split by blanks in -E output, and
! no blanks are inserted into character literals or Hollerith constants.
! The statements are indented so that their original columns lie beyond
! column 7, where -E output would otherwise restore them.
#define D_OR_S() d
#define STAR() *
#define ID(x) x
#define X() x
#define FMT() format
subroutine s(n, a, i, l)
  integer :: n, i, nn
  real :: a(n)
  logical :: l
!CHECK: i = n **           2
!UNPARSE: i=n**2_4
          i = n STAR()STAR() 2
!CHECK: l = n == 1
!UNPARSE: l=n==1_4
          l = n ID(=)= 1
!CHECK: l = n .eq. 1
!UNPARSE: l=n==1_4
          l = n .ID(eq). 1
!CHECK: a(1) = 1.e5
!UNPARSE: a(1_8)=1.e5_4
          a(1) = 1.ID()e5
!CHECK: nn = nn
!UNPARSE: nn=nn
          nn = ID(n)ID(n)
!CHECK: call dfoo        ( n , i       )
!UNPARSE: CALL dfoo(n,i)
          call D_OR_S()foo ( n , ID( i ) )
!CHECK: call x('a b')
!UNPARSE: CALL x("a b")
          call X()('a b')
!CHECK: call x(4Hab c)
!UNPARSE: CALL x("ab c    ")
          call X()(4Hab c)
!CHECK: call x(3Hé b)
!UNPARSE: CALL x("\303\251 b    ")
          call X()(3Hé b)
!CHECK: call x('a b')
!UNPARSE: CALL x("a b")
          call x(/* c */'a b')
!CHECK: 10        format(1X3Ha b)
!UNPARSE: 10 FORMAT(1X,"a b")
10        FMT()(1X3Ha b)
!CHECK: call x(3Ha é)
          call X()(3Ha é)
!CHECK: i = n     + 2
          i = ID(n) + 2
!CHECK: i = n      + 2
          i = ID(n)  + 2
!CHECK: !$omp parallel private(dx)
!$omp parallel private(D_OR_S()x)
!$omp end parallel
end
