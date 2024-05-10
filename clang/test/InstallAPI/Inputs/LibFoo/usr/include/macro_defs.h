#ifndef MACRO_DEFS_H
#define MACRO_DEFS_H 

#if defined(NONDarwin) 
  #define LINUX "$linux"
  #define DARWIN 
#elif defined(Darwin) 
  #define LINUX 
  #define DARWIN "$darwin" 
#else 
  #define LINUX 
  #define DARWIN 
#endif 

#endif // MACRO_DEFS_H
