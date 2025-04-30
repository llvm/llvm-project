#define UNSIGNED
#define EXTEND(S,D)	zapnot S, 15, D
#define __divl		__divlu
#include <divl.S>
