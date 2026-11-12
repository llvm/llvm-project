! Every -load plugin is loaded, not only the last: here the plugin that
! -plugin runs is the first of two.

! REQUIRES: plugins, examples
! XFAIL: system-aix

! RUN: %flang_fc1 -load %llvmshlibdir/flangPrintFunctionNames%pluginext -load %llvmshlibdir/flangOmpReport%pluginext -plugin print-fns %s 2>&1 | FileCheck %s

! CHECK: Subroutine: s
! CHECK: ==== Subroutines: 1 ====

subroutine s
end subroutine
