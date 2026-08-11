!RUN: not %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s
! CHECK: error: Could not parse
! CHECK: 4:21: error: expected '('
program p; function fn(); end; end;
