// REQUIRES: target={{.*}}-{{darwin|macos}}{{.*}}
// RUN: %clang_cc1 -fblocks -triple arm64-apple-darwin -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fblocks -x objective-c++ -triple arm64-apple-darwin -emit-llvm -o - %s | FileCheck %s

typedef struct
{
  int a;
} BlockState;

typedef void(^InBlock)(int); 

void sink(InBlock);

int doMagic(void)
{
  __block BlockState state;
  sink(^(int in) {
    state.a += in;
  });
  return state.a;
}

// block layout in the regular c-string section
// CHECK: @OBJC_LAYOUT_BITMAP_{{.*}} = private unnamed_addr constant {{.*}} section "__TEXT,__cstring,cstring_literals"
