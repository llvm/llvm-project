
#define SMALL_BOUND 0x1.921fb4p+16f

#if defined EXTRA_PRECISION
extern int MATH_PRIVATE(trigredsmall)(__private float *r, __private float *rr, float x);
extern int MATH_PRIVATE(trigredlarge)(__private float *r, __private float *rr, float x);
extern int MATH_PRIVATE(trigred)(__private float *r, __private float *rr, float x);
#else
extern int MATH_PRIVATE(trigredsmall)(__private float *r, float x);
extern int MATH_PRIVATE(trigredlarge)(__private float *r, float x);
extern int MATH_PRIVATE(trigred)(__private float *r, float x);
#endif

extern float MATH_PRIVATE(sincosred2)(float x, float y, __private float *cp);

extern float MATH_PRIVATE(sincosred)(float x, __private float *cp);

extern CONSTATTR float MATH_PRIVATE(tanred)(float x, int regn);

