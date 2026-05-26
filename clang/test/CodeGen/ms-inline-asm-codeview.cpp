// RUN: %clang_cc1 -triple i386-pc-windows-msvc -fasm-blocks -gcodeview \
// RUN:   -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

#line 100 "t.cpp"
void f() {
  __asm {
    nop
    nop
  }
}

// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: !srcloc ![[SRCLOC:[0-9]+]]
// CHECK: ![[SRCLOC]] = !{i64 {{[0-9]+}}, i64 {{[0-9]+}}, ![[DBGLOCS:[0-9]+]]}
// CHECK: ![[DBGLOCS]] = !{!"inlineasm.dbg.line", i32 102, i32 5, i32 103, i32 5}
