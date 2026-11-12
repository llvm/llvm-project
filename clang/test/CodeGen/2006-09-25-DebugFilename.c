// RUN: not %clang_cc1  %s -emit-llvm -o /dev/null
#include "2006-09-25-DebugFilename.h"
int func1() { return hfunc1(); }
int func2() { fluffy; return hfunc1(); } // expected-error {{use of undeclared identifier 'fluffy'}}
