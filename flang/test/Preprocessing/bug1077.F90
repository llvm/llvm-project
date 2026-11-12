!RUN: %flang -E %s 2>&1 | FileCheck %s
!CHECK: print *,((1)+(2)),4
#define foo(x,y) ((x)+(y))
print *,&
foo(1,2)&
,4
end
