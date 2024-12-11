#include <ptrcheck.h>

//strict-note@+1{{passing argument to parameter 'foo' here}}
void funcWithAnnotation(char *__sized_by(4) foo, char *__sized_by(5) bar);


#pragma clang system_header

void funcInSDK(char *ptr, char * __bidi_indexable bidi) {
  //strict-error@+1{{passing 'char *' to parameter of incompatible type 'char *__single __sized_by(4)' (aka 'char *__single') casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  funcWithAnnotation(ptr, bidi);
}

