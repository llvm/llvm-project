! RUN: %flang_fc1 -fsyntax-only -pedantic %s 2>&1 | FileCheck --allow-empty %s
! CHECK-NOT: error:
! Regression test for https://github.com/llvm/llvm-project/issues/124191
character(3) :: arr(5) = ['aa.', 'bb.', 'cc.', 'dd.', 'ee.']
arr([(mod(iachar(arr(i:i-1:-1)(1:1)),5)+1, i=2,5,3)]) = arr(5:2:-1)
end
