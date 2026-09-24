!RUN: %flang -E -DNVAR=2+1+0+0 %s 2>&1 | FileCheck %s
!CHECK: pass
#if NVAR > 2
call pass
#endif
end
