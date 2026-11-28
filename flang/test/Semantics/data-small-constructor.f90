! RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

subroutine small_complex
  type pair
    complex values(-1:1)
  end type
  type(pair) x
  data x / pair((1.0, 2.0)) /
  ! CHECK: x (InDataStmt){{.*}}init:pair(values=[COMPLEX(4)::(1._4,2._4),(1._4,2._4),(1._4,2._4)])
end
