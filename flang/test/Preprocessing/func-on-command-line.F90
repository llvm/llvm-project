! RUN: %flang_fc1 -fdebug-unparse "-Dfoo(a,b)=bar(a+b)" %s | FileCheck %s
! CHECK: CALL bar(3_4)
call foo(1,2)
end
