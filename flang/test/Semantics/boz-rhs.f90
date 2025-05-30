! RUN: %flang_fc1 -fdebug-unparse %s | FileCheck %s
double precision dp
integer(8) i64
!CHECK: dp=1._8
dp = z'3ff0000000000000'
!CHECK: i64=-77129852189294865_8
i64 = z'feedfacedeadbeef'
end
