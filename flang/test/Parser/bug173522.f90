!RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
!CHECK: 3:10: error: invalid or unknown I/O control specification
write (*,nml=123)
end
