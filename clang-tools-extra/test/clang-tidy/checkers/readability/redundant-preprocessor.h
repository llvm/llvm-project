#ifndef FOO
#ifndef FOO // this would warn, but not in a header
void f();
#endif
#endif
