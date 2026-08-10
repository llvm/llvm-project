// RUN: %clang_cc1 -triple i386-apple-darwin9 %s -emit-llvm -o - | FileCheck -check-prefix CHECK-X32 %s
// CHECK-X32: %union.command = type <{ ptr, [2 x i8] }>

// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -emit-llvm -o - | FileCheck -check-prefix CHECK-X64 %s
// CHECK-X64: %union.command = type <{ ptr, [2 x i8] }>

#pragma pack(push, 2)
typedef union command {
  void *windowRef;
  struct menu {
    void *menuRef;
    unsigned char menuItemIndex;
  } menu;
} command;

command c;
