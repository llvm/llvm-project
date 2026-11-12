// RUN: %clang -arch x86_64 %s -fsyntax-only -Xclang -print-stats
// REQUIRES: macos-sdk-10.12
#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#endif

