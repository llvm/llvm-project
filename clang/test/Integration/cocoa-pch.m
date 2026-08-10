// RUN: %clang -arch x86_64 -x objective-c-header %s -o %t.h.pch
// RUN: touch %t.empty.m
// RUN: %clang -arch x86_64 -fsyntax-only %t.empty.m -include %t.h -Xclang -ast-dump 2>&1 > /dev/null
// REQUIRES: macos-sdk-10.12
#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#endif

