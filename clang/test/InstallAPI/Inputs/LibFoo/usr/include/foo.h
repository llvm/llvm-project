#ifndef FOO_H
#define FOO_H 
#include <macro_defs.h> 

#if defined(Foo) 
  #define FOO "FooLib$" 
#else 
  #define FOO 
#endif 

#define __STRING(x)     #x
#define PLATFORM_ALIAS(sym)	__asm("_" FOO __STRING(sym) DARWIN LINUX)
extern int foo() PLATFORM_ALIAS(foo);

#endif 
