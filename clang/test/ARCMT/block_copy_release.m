// RUN: %clang_cc1 -fblocks -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fblocks -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result

#include "Common.h"

typedef void (^blk)(int);

void func(blk b) {
  blk c = Block_copy(b);
  Block_release(c);
}

void func2(id b) {
  id c = Block_copy(b);
  Block_release(c);
}
