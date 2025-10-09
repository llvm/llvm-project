! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! Ensure that DATA-like default component initializers work.
! CHECK: j (InDataStmt) size=4 offset=0: ObjectEntity type: INTEGER(4) init:123_4
type t
  integer j/123/
end type
end
