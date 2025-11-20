! RUN: %flang -E %s | FileCheck %s
#define KWM a
#define FLM(x) b FLM2(x) KWM c
#define FLM2(x) d FLM(x) e
! CHECK: a
KWM
! CHECK: b d FLM(y) e a c
FLM(y)
! CHECK: b d FLM(a) e a c
FLM(KWM)
! CHECK: b d FLM(b d FLM(y) e a c) e a c
FLM(FLM(y))
! CHECK: b d FLM(b d FLM(a) e a c) e a c
FLM(FLM(KWM))
