!RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
!CHECK: warning: overflow on REAL(4) to REAL(2) conversion after folding a call to 'exp' [-Wfolding-exception]
print *, exp((11.265625_2,1._2))
end
