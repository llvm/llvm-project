/* Defining _LIBC_TEST stops fpu_control.h from defining the
   hard-float versions of macros (for use with dynamic VFP detection)
   when compiling for soft-float.  */
#define _LIBC_TEST
#include <math/test-fpucw.c>
