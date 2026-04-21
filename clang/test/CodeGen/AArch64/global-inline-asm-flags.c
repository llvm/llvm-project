// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +pauth -flto=thin -emit-llvm -o - %s | FileCheck %s
// REQUIRES: aarch64-registered-target

asm (
    ".text" "\n"
    ".balign 16" "\n"
    ".globl foo\n"
    "foo:\n"
    "pacib     x30, x27" "\n"
    "retab" "\n"
    ".symver foo, foo@VER" "\n"
    ".symver foo, foo@ANOTHERVER" "\n"
    ".globl bar\n"
    "bar:\n"
    "pacib     x30, x27" "\n"
    "retab" "\n"
    ".symver bar, bar@VER" "\n"
    ".previous" "\n"
);

// CHECK: module asm ".text"
// CHECK: module asm ".balign 16"
// CHECK: module asm ".globl foo"
// CHECK: module asm "foo:"
// CHECK: module asm "pacib     x30, x27"
// CHECK: module asm "retab"
// CHECK: module asm ".symver foo, foo@VER"
// CHECK: module asm ".symver foo, foo@ANOTHERVER"
// CHECK: module asm ".globl bar"
// CHECK: module asm "bar:"
// CHECK: module asm "pacib     x30, x27"
// CHECK: module asm "retab"
// CHECK: module asm ".symver bar, bar@VER"
// CHECK: module asm ".previous"

// CHECK: !{{.*}} = !{i32 6, !"global-asm-symbols", ![[SYM:[0-9]+]]}
// CHECK: ![[SYM]] = !{![[SBAR1:[0-9]+]], ![[SBAR2:[0-9]+]], ![[SBAR3:[0-9]+]], ![[SFOO1:[0-9]+]], ![[SFOO2:[0-9]+]]}
// CHECK: ![[SBAR1]] = !{!"bar", i32 2050}
// CHECK: ![[SBAR2]] = !{!"bar@VER", i32 2050}
// CHECK: ![[SBAR3]] = !{!"foo@ANOTHERVER", i32 2050}
// CHECK: ![[SFOO1]] = !{!"foo", i32 2050}
// CHECK: ![[SFOO2]] = !{!"foo@VER", i32 2050}
// CHECK: !{{.*}} = !{i32 6, !"global-asm-symvers", ![[SYMVER:[0-9]+]]}
// CHECK: ![[SYMVER]] = !{![[VFOO:[0-9]+]], ![[VBAR:[0-9]+]]}
// CHECK: ![[VFOO:[0-9]+]] = !{!"foo", !"foo@VER", !"foo@ANOTHERVER"}
// CHECK: ![[VBAR:[0-9]+]] = !{!"bar", !"bar@VER"}
