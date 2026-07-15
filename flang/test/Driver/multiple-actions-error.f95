! Verify that the frontend driver raises the expected error when multiple
! actions are specified.
!
! RUN: not %flang_fc1 -fsyntax-only -fsyntax-only %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=ERROR,ACTIONS-1
!
! RUN: not %flang_fc1 -E -fsyntax-only %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=ERROR,ACTIONS-2
!
! RUN: not %flang_fc1 -fsyntax-only -E -emit-llvm %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=ERROR,ACTIONS-3
!
! If one or more options are specified with -Xflang, they will appear last in
! the error message.
!
! RUN: not %flang -S -Xflang -emit-llvm %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=ERROR,ACTIONS-4
!
! RUN: not %flang -Xflang -emit-llvm -S %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=ERROR,ACTIONS-4
!
! RUN: not %flang -Xflang -emit-obj -S -Xflang -emit-llvm %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=ERROR,ACTIONS-5
!
! ERROR: error: only one action option is allowed.
! ACTIONS-1: Got '-fsyntax-only', '-fsyntax-only'
! ACTIONS-2: Got '-E', '-fsyntax-only'
! ACTIONS-3: Got '-fsyntax-only', '-E', '-emit-llvm'
! ACTIONS-4: Got '-S', '-emit-llvm'
! ACTIONS-5: Got '-S', '-emit-obj', '-emit-llvm'
