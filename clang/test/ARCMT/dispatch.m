// RUN: %clang_cc1 -fblocks -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fblocks -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result

#include "Common.h"

dispatch_object_t getme(void);

void func(dispatch_object_t o) {
  dispatch_retain(o);
  dispatch_release(o);
  dispatch_retain(getme());
}

void func2(xpc_object_t o) {
  xpc_retain(o);
  xpc_release(o);
}
