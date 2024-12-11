#include <ptrcheck.h>

#pragma clang system_header

// strict-note@+1{{passing argument to parameter 'foo' here}}
void funcWithAnnotation(int *__sized_by(4) foo);
void funcWithoutAnnotation(int * foo);
extern int * __single safeGlobal;
extern int * unsafeGlobal;
