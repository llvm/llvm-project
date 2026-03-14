! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK: 3:13: error: expected end of statement
common/blk/a,,b
end
