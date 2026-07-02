! RUN: %flang_fc1 -fsyntax-only %s 2>&1
! RUN: %flang -E %s 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -pedantic -Werror -fdebug-dump-parse-tree-no-sema \
! RUN:        -DTEST_ERRORS=1 %s 2>&1 | FileCheck %s --check-prefix=ERROR

      integer :: i

! CHECK-NOT: C comment
! CHECK: i = 1
! ERROR: portability: nonstandard usage: C-style comment
/* Old style C comments
 *comments
 */
#define VAL 1

! ERROR: portability: nonstandard usage: C-style comment
/* Single-line C comment */
 /* Single-line C comment */
  /* Single-line C comment */
   /* Single-line C comment */
    /* Single-line C comment */
     /* Single-line C comment */
      /* Single-line C comment */
       /* Single-line C comment */
      i = VAL

! CHECK-NOT: /*
! CHECK-NOT: Multi
! CHECK-NOT: C comment
! CHECK-NOT: */
! CHECK: i = 2
/*
 * Multi-line C comment
 * comment
 */
  /* Multi-line
C comment */
      i = 2

! CHECK: i = 3
! CHECK: i = 4
! CHECK: 9 i = 5+ 6
/* C comment + statement */ i = 3
      /* C comment + statement */ i = 4
/**/90i = 5
/**/ + + 6

! CHECK-NOT: /*
! CHECK-NOT: C comment
! CHECK-NOT: */
! CHECK: 100 i = 7
/* Multi-line C comment + statement
*/100 i = 7

! CHECK: i = 8
! CHECK-NOT: C comment
      i = 8 /* C comment after statement */

! CHECK: print
! CHECK-NOT: C comment
	print *, "TAB"
	/* C comment after tab */

! ERROR: error: Character in fixed-form label field must be a digit
#if TEST_ERRORS
/* Not-terminated C comment
#endif
      end
