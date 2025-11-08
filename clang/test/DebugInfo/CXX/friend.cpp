// RUN: %clang -emit-llvm -S -g %s -o - | FileCheck %s

class MyFriend;

class SomeClass {
  friend class MyFriend;
  typedef int SomeType;
};

SomeClass *x;

struct MyFriend {
  static void func(SomeClass::SomeType) {
  }
};

// Emitting debug info for friends unnecessarily bloats debug info without any
// known benefit or debugger feature that requires it. Re-enable this is a
// use-case appears.
// CHECK-NOT: DW_TAG_friend
