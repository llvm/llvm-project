! RUN: %flang -E %s 2>&1 | FileCheck %s

! CHECK: print *, 'pass 1'
#define IS_DEFINED
#define M1 defined(IS_DEFINED)
#if M1
print *, 'pass 1'
#else
print *, 'fail 1'
#endif

! CHECK: print *, 'pass 2'
#define M2 defined IS_DEFINED
#if M2
print *, 'pass 2'
#else
print *, 'fail 2'
#endif

! CHECK: print *, 'pass 3'
#define M3 defined(IS_UNDEFINED)
#if M3
print *, 'fail 3'
#else
print *, 'pass 3'
#endif

! CHECK: print *, 'pass 4'
#define M4 defined IS_UNDEFINED
#if M4
print *, 'fail 4'
#else
print *, 'pass 4'
#endif

! CHECK: print *, 'pass 5'
#define DEFINED_KEYWORD defined
#define M5(x) DEFINED_KEYWORD(x)
#define KWM1 1
#if M5(KWM1)
print *, 'pass 5'
#else
print *, 'fail 5'
#endif

! CHECK: print *, 'pass 6'
#define KWM2 KWM1
#if M5(KWM2)
print *, 'pass 6'
#else
print *, 'fail 6'
#endif

! CHECK: print *, 'pass 7'
#if M5(IS_UNDEFINED)
print *, 'fail 7'
#else
print *, 'pass 7'
#endif

! CHECK: print *, 'pass 8'
#define KWM3 IS_UNDEFINED
#if M5(KWM3)
print *, 'pass 8'
#else
print *, 'fail 8'
#endif

! CHECK: print *, 'pass 9'
#define M6(x) defined(x)
#if M6(KWM1)
print *, 'pass 9'
#else
print *, 'fail 9'
#endif

! CHECK: print *, 'pass 10'
#if M6(KWM2)
print *, 'pass 10'
#else
print *, 'fail 10'
#endif

! CHECK: print *, 'pass 11'
#if M6(IS_UNDEFINED)
print *, 'fail 11'
#else
print *, 'pass 11'
#endif

! CHECK: print *, 'pass 12'
#if M6(KWM3)
print *, 'pass 12'
#else
print *, 'fail 12'
#endif

! CHECK: print *, 'pass 13'
#define M7(A, B) ((A) * 10000 + (B) * 100)
#define M8(A, B, C, AA, BB) ( \
  (defined(AA) && defined(BB)) && \
  (M7(A, B) C M7(AA, BB)))
#if M8(9, 5, >, BAZ, FUX)
print *, 'fail 13'
#else
print *, 'pass 13'
#endif

! CHECK: print *, 'pass 14'
#define M9() (defined(IS_UNDEFINED))
#if M9()
print *, 'fail 14'
#else
print *, 'pass 14'
#endif

end
