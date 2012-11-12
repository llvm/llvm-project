// RUN: %clang -arch x86_64 %s -fsyntax-only -Xclang -print-stats 
#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#endif

