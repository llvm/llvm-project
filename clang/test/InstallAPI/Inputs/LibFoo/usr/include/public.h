#ifndef PUBLIC_H
#define PUBLIC_H 
#include <macro_defs.h>

#define __STRING(x)     #x
#define PLATFORM_ALIAS(sym)	__asm("_" __STRING(sym) DARWIN LINUX)
extern int foo() PLATFORM_ALIAS(foo);

#endif 
