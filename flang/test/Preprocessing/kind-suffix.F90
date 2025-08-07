! RUN: %flang -E %s 2>&1 | FileCheck %s
#define n k
#define _m _p
parameter(n=4)
!CHECK: print *,1_k
print *,1_n
!CHECK: print *,1_p
print *,1_m
end
