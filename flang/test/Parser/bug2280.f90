!RUN: %flang -fc1 -fdebug-unparse %s | FileCheck %s
!CHECK: 1 FORMAT(1X)
1 format(1x)
!CHECK: 2 FORMAT(1X)
2 format(x)
end
