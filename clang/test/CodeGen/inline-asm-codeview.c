// RUN: %clang_cc1 -triple i386-pc-windows-msvc -gcodeview \
// RUN:   -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-pc-windows-msvc -gcodeview \
// RUN:   -debug-info-kind=limited -S -o - %s | FileCheck %s --check-prefix=ASM
// RUN: %clang_cc1 -triple i386-pc-windows-msvc -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=NO-DEBUG --implicit-check-not=inlineasm.dbg.offset

// The concatenated template has no newlines. Each instruction nevertheless
// belongs to a different source line. Named operands, %% and $ change length
// when converted to an LLVM asm template.
#line 100 "inline-asm-codeview.c"
void gnu(int x) {
  asm volatile(
      "movl %[value], %%eax;"
      "addl $1, %%eax;"
      "incl %%eax"
      :
      : [value] "r"(x)
      : "eax", "cc");
}

// ASM-LABEL: _gnu:
// ASM: #APP
// ASM: .cv_loc 0 1 102 8
// ASM-NEXT: movl
// ASM-NEXT: .cv_loc 0 1 103 8
// ASM-NEXT: addl $1, %eax
// ASM-NEXT: .cv_loc 0 1 104 8
// ASM-NEXT: incl %eax
// ASM: #NO_APP

// CHECK: call void asm sideeffect "movl $0, %eax;addl $$1, %eax;incl %eax"
// CHECK-SAME: !srcloc ![[SRC:[0-9]+]]
// NO-DEBUG: call void asm sideeffect
// NO-DEBUG-SAME: !srcloc

// Cover dialect alternatives in the frontend mapping.
#line 200 "inline-asm-codeview.c"
void variants(int x) {
  asm volatile(
      "{incl %0|inc %0};"
      "nop"
      : "+r"(x));
}

// CHECK: call i32 asm sideeffect "$(incl $0$|inc $0$);nop"
// CHECK-SAME: !srcloc ![[VAR_SRC:[0-9]+]]

// Basic GNU asm also distinguishes instructions on the same source line.
#line 300 "inline-asm-codeview.c"
void basic(void) {
  asm("nop;nop");
}

// CHECK: call void asm sideeffect "nop;nop"
// CHECK-SAME: !srcloc ![[BASIC_SRC:[0-9]+]]
// ASM-LABEL: _basic:
// ASM: #APP
// ASM: .cv_loc 2 1 301 8
// ASM-NEXT: nop
// ASM-NEXT: .cv_loc 2 1 301 12
// ASM-NEXT: nop
// ASM: #NO_APP

// CHECK: ![[SRC]] = !{i64 {{[0-9]+}}, ![[LOCS:[0-9]+]]}
// CHECK: ![[LOCS]] = !{!"inlineasm.dbg.offset", i32 0, i32 102, i32 8,
// CHECK-SAME: i32 14, i32 103, i32 8,
// CHECK-SAME: i32 29, i32 104, i32 8,
// CHECK: ![[VAR_SRC]] = !{i64 {{[0-9]+}}, ![[VAR_LOCS:[0-9]+]]}
// CHECK: ![[VAR_LOCS]] = !{!"inlineasm.dbg.offset",
// CHECK-SAME: i32 2, i32 202, i32 9,
// CHECK-SAME: i32 11, i32 202, i32 17,
// CHECK-SAME: i32 20, i32 203, i32 8}
// CHECK: ![[BASIC_SRC]] = !{i64 {{[0-9]+}}, ![[BASIC_LOCS:[0-9]+]]}
// CHECK: ![[BASIC_LOCS]] = !{!"inlineasm.dbg.offset", i32 0, i32 301, i32 8, i32 3, i32 301, i32 11, i32 4, i32 301, i32 12}
