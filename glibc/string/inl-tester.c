/* We want to test the inline functions here.  */

#define DO_STRING_INLINES
#undef __USE_STRING_INLINES
#define __USE_STRING_INLINES 1
#include "tester.c"
