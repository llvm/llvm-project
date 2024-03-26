! RUN: %flang -E %s | FileCheck %s
#define A B(c)
#define B(d) d); call E(d
#define E(f) G(f)
!CHECK: call I(c); call G(c)
call I(A)
end
