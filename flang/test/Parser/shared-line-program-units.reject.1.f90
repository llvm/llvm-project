!RUN: not %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s
! CHECK: error: Could not parse
! CHECK: 4:36: error: end of file
function fn(); end; function fn2();
