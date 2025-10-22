! Verify that`-O{n}` is indeed taken into account when defining the LLVM optimization/middle-end pass pipeline.

! RUN: %flang -S -O0 %s -Xflang -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O0
! RUN: %flang -S -O0 %s -flto -Xflang -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O0-ANYLTO
! RUN: %flang -S -O0 %s -flto=thin -Xflang -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O0-ANYLTO
! RUN: %flang_fc1 -S -O0 %s -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O0
! RUN: %flang_fc1 -S -O0 %s -flto=full -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O0-ANYLTO
! RUN: %flang_fc1 -S -O0 %s -flto=thin -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O0-ANYLTO

! RUN: %flang -S -O2 %s -Xflang -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O2
! RUN: %flang -S -O2 %s -flto -Xflang -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O2-LTO
! RUN: %flang -S -O2 %s -flto=thin -Xflang -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O2-THINLTO
! RUN: %flang_fc1 -S -O2 %s -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O2
! RUN: %flang_fc1 -S -O2 %s -flto=full -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O2-LTO
! RUN: %flang_fc1 -S -O2 %s -flto=thin -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O2-THINLTO

! Verify that only the right-most `-O{n}` is used
! RUN: %flang -S -O2 -O0 %s -Xflang -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O0
! RUN: %flang_fc1 -S -O2 -O0 %s -fdebug-pass-manager -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O0

! Verify that passing -Os/-Oz have the desired effect on the pass pipelines.
! RUN: %flang -S -Os %s -Xflang -fdebug-pass-manager -o /dev/null 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK-OSIZE
! RUN: %flang -S -Oz %s -Xflang -fdebug-pass-manager -o /dev/null 2>&1 \
! RUN:     | FileCheck %s --check-prefix=CHECK-OSIZE

! CHECK-O0-NOT: Running pass: SimplifyCFGPass on simple_loop_
! CHECK-O0: Running analysis: TargetLibraryAnalysis on simple_loop_
! CHECK-O0-ANYLTO: Running pass: CanonicalizeAliasesPass on [module]
! CHECK-O0-ANYLTO: Running pass: NameAnonGlobalPass on [module]

! CHECK-O2: Running pass: SimplifyCFGPass on simple_loop_

! CHECK-O2-LTO-NOT: Running pass: EliminateAvailableExternallyPass
! CHECK-O2-LTO: Running pass: CanonicalizeAliasesPass on [module]
! CHECK-O2-LTO: Running pass: NameAnonGlobalPass on [module]

! CHECK-O2-THINLTO-NOT: Running pass: LoopVectorizePass
! CHECK-O2-THINLTO: Running pass: CanonicalizeAliasesPass on [module]
! CHECK-O2-THINLTO: Running pass: NameAnonGlobalPass on [module]

! -Os/-Oz imply -O2, so check that a pass that runs on O2 is run. Then check
! that passes like LibShrinkWrap, that should not be run when optimizing for
! size, are not run (see llvm/lib/Passes/PassBuilderPipelines.cpp).
! CHECK-OSIZE: Running pass: SimplifyCFGPass on simple_loop_
! CHECK-OSIZE-NOT: Running pass: LibCallsShrinkWrapPass on simple_loop_

subroutine simple_loop
  integer :: i
  do i=1,5
  end do
end subroutine
