// RUN: %clang_cc1 -triple i386-unknown-unknown -O3 %s -emit-llvm -o - | FileCheck %s

int foo(int i) {
  int j = 0;
  switch (i) {
  case -1:
    j = 1; break;
  case 1 :
    j = 2; break;
  case 2:
    j = 3; break;
  default:
    j = 42; break;
  }
  j = j + 1;
  return j;
}

int foo2(int i) {
  int j = 0;
  switch (i) {
  case 1 :
    j = 2; break;
  case 2 ... 10:
    j = 3; break;
  default:
    j = 42; break;
  }
  j = j + 1;
  return j;
}

int foo3(int i) {
  int j = 0;
  switch (i) {
  default:
    j = 42; break;
  case 111:
    j = 111; break;
  case 0 ... 100:
    j = 1; break;
  case 222:
    j = 222; break;
  }
  return j;
}


static int foo4(int i) {
  int j = 0;
  switch (i) {
  case 111:
    j = 111; break;
  case 0 ... 100:
    j = 1; break;
  case 222:
    j = 222; break;
  default:
    j = 42; break;
  case 501 ... 600:
    j = 5; break;
  }
  return j;
}

// CHECK-LABEL: define{{.*}} i32 @foo4t()
// CHECK: ret i32 376
// CHECK: }
int foo4t() {
  // 111 + 1 + 222 + 42 = 376
  return foo4(111) + foo4(99) + foo4(222) + foo4(601);
}

// CHECK-LABEL: define{{.*}} void @foo5()
// CHECK-NOT: switch
// CHECK: }
void foo5(){
    switch(0){
    default:
        if (0) {

        }
    }
}

// CHECK-LABEL: define{{.*}} void @foo6()
// CHECK-NOT: switch
// CHECK: }
void foo6(){
    switch(0){
    }
}

// CHECK-LABEL: define{{.*}} void @foo7()
// CHECK-NOT: switch
// CHECK: }
void foo7(){
    switch(0){
      foo7();
    }
}


// CHECK-LABEL: define{{.*}} i32 @f8(
// CHECK: ret i32 3
// CHECK: }
int f8(unsigned x) {
  switch(x) {
  default:
    return 3;
  case 0xFFFFFFFF ... 1: // This range should be empty because x is unsigned.
    return 0;
  }
}

// Ensure that default after a case range is not ignored.
//
// CHECK-LABEL: define{{.*}} i32 @f9()
// CHECK: ret i32 10
// CHECK: }
static int f9_0(unsigned x) {
  switch(x) {
  case 10 ... 0xFFFFFFFF:
    return 0;
  default:
    return 10;
  }
}
int f9() {
  return f9_0(2);
}

// Ensure that this doesn't compile to infinite loop in g() due to
// miscompilation of fallthrough from default to a (tested) case
// range.
//
// CHECK-LABEL: define{{.*}} i32 @f10()
// CHECK: ret i32 10
// CHECK: }
static int f10_0(unsigned x) {
  switch(x) {
  default:
    x += 1;
  case 10 ... 0xFFFFFFFF:
    return 0;
  }
}

int f10() {
  f10_0(1);
  return 10;
}

// This generated incorrect code because of poor switch chaining.
//
// CHECK-LABEL: define{{.*}} i32 @f11(
// CHECK: ret i32 3
// CHECK: }
int f11(int x) {
  switch(x) {
  default:
    return 3;
  case 10 ... 0xFFFFFFFF:
    return 0;
  }
}

// This just asserted because of the way case ranges were calculated.
//
// CHECK-LABEL: define{{.*}} i32 @f12(
// CHECK: ret i32 3
// CHECK: }
int f12(int x) {
  switch (x) {
  default:
    return 3;
  case 10 ... -1: 
    return 0;
  }
}

// Make sure return is not constant (if empty range is skipped or miscompiled)
//
// CHECK-LABEL: define{{.*}} i32 @f13(
// CHECK: ret i32 %
// CHECK: }
int f13(unsigned x) {
  switch(x) {
  case 2:
    // fallthrough empty range
  case 10 ... 9:
    return 10;
  default:
    return 0;
  }
}

// Don't delete a basic block that we want to introduce later references to.
// This isn't really specific to switches, but it's easy to show with them.
// rdar://problem/8837067
int f14(int x) {
  switch (x) {

  // case range so that the case block has no predecessors
  case 0 ... 15:
    // any expression which doesn't introduce a new block
    (void) 0;
    // kaboom

  default:
    return x;
  }
}
