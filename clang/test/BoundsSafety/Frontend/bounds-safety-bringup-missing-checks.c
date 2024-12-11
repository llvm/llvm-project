

// RUN: not %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=none -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=POS_NONE
// RUN: not %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=none,compound_literal_init,all -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=POS_NONE

// POS_NONE: invalid value 'none' in '-fbounds-safety-bringup-missing-checks='; did you mean '-fno-bounds-safety-bringup-missing-checks'?

// RUN: not %clang_cc1 -fbounds-safety -fno-bounds-safety-bringup-missing-checks=none -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=NEG_NONE
// RUN: not %clang_cc1 -fbounds-safety -fno-bounds-safety-bringup-missing-checks=none,compound_literal_init,all -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=NEG_NONE

// NEG_NONE: invalid value 'none' in '-fno-bounds-safety-bringup-missing-checks='; did you mean '-fbounds-safety-bringup-missing-checks'?

// RUN: not %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=abc -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=POS_ABC
// RUN: not %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=abc,compound_literal_init,all -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=POS_ABC

// POS_ABC: invalid value 'abc' in '-fbounds-safety-bringup-missing-checks='


// RUN: not %clang_cc1 -fbounds-safety -fno-bounds-safety-bringup-missing-checks=abc -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=NEG_ABC
// RUN: not %clang_cc1 -fbounds-safety -fno-bounds-safety-bringup-missing-checks=abc,compound_literal_init,all -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=NEG_ABC

// NEG_ABC: invalid value 'abc' in '-fno-bounds-safety-bringup-missing-checks='
