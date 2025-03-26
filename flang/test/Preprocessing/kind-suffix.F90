! RUN: %flang -E %s 2>&1 | FileCheck %s
#define n k
parameter(n=4)
!CHECK: print *,1_k
print *,1_n
end
