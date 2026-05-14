! Test that the deprecated -Wopen-mp-* and -Wopen-acc-* spellings produce
! deprecation warnings and suggest the canonical -Wopenmp-* / -Wopenacc-*
! spellings.

! RUN: %flang_fc1 -fsyntax-only -Wopen-mp-usage %s 2>&1 | FileCheck --check-prefix=DEPRECATED %s
! RUN: %flang_fc1 -fsyntax-only -Wno-open-mp-usage %s 2>&1 | FileCheck --check-prefix=DEPRECATED-NO %s
! RUN: %flang_fc1 -fsyntax-only -Wopen-acc-usage %s 2>&1 | FileCheck --check-prefix=DEPRECATED-ACC %s
! RUN: %flang_fc1 -fsyntax-only -Wopenmp-usage %s 2>&1 | FileCheck --allow-empty --check-prefix=CANONICAL %s
! RUN: %flang_fc1 -fsyntax-only -Wopenacc-usage %s 2>&1 | FileCheck --allow-empty --check-prefix=CANONICAL %s

! RUN: %flang -fsyntax-only -Wopen-mp-usage %s 2>&1 | FileCheck --check-prefix=DEPRECATED %s
! RUN: %flang -fsyntax-only -Wno-open-mp-usage %s 2>&1 | FileCheck --check-prefix=DEPRECATED-NO %s
! RUN: %flang -fsyntax-only -Wopen-acc-usage %s 2>&1 | FileCheck --check-prefix=DEPRECATED-ACC %s
! RUN: %flang -fsyntax-only -Wopenmp-usage %s 2>&1 | FileCheck --allow-empty --check-prefix=CANONICAL %s
! RUN: %flang -fsyntax-only -Wopenacc-usage %s 2>&1 | FileCheck --allow-empty --check-prefix=CANONICAL %s

! DEPRECATED: warning: -Wopen-mp-usage is deprecated; use -Wopenmp-usage instead
! DEPRECATED-NO: warning: -Wno-open-mp-usage is deprecated; use -Wno-openmp-usage instead
! DEPRECATED-ACC: warning: -Wopen-acc-usage is deprecated; use -Wopenacc-usage instead
! CANONICAL-NOT: deprecated

program test
end program test
