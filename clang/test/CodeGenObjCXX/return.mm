/// -fstrict-return is the default.
// RUN: %clang_cc1 -Wno-error=return-type -emit-llvm -fblocks -triple x86_64-apple-darwin -o - %s | FileCheck %s
// RUN: %clang_cc1 -Wno-error=return-type -emit-llvm -fblocks -triple x86_64-apple-darwin -O -o - %s | FileCheck %s

@interface I
@end

@implementation I

- (int)method {
}

@end

enum Enum {
  a
};

int (^block)(Enum) = ^int(Enum e) {
  switch (e) {
  case a:
    return 1;
  }
};

// Ensure that both methods and blocks don't use the -fstrict-return undefined
// behaviour optimization.

// CHECK-NOT: call void @llvm.trap
// CHECK-NOT: unreachable
