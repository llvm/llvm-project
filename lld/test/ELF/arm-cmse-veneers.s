// REQUIRES: arm
/// Test that symbol visibilities of <sym, __acle_se_sym> pairs in the objects
/// are preserved in the executable.

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.main %s -I %S/Inputs -o %t.o
// RUN: ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 %t.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
// RUN: llvm-readelf -s %t | FileCheck %s --check-prefixes=SYM

// CHECK: Disassembly of section .gnu.sgstubs:

// CHECK-LABEL: <global_foo>:
// CHECK-NEXT:    20000: sg
// CHECK-NEXT:           b.w {{.*}} <__acle_se_global_foo>
// CHECK-EMPTY:
// CHECK-LABEL: <weak_bar>:
// CHECK-NEXT:    20008: sg
// CHECK-NEXT:           b.w {{.*}} <__acle_se_weak_bar>
// CHECK-EMPTY:
// CHECK-LABEL: <global_baz>:
// CHECK-NEXT:    20010: sg
// CHECK-NEXT:           b.w {{.*}} <__acle_se_global_baz>
// CHECK-EMPTY:
// CHECK-LABEL: <weak_qux>:
// CHECK-NEXT:    20018: sg
// CHECK-NEXT:           b.w {{.*}} <__acle_se_weak_qux>

// SYM: 00020001 {{.*}} GLOBAL {{.*}}           global_foo
// SYM: 00008001 {{.*}} GLOBAL {{.*}} __acle_se_global_foo
// SYM: 00020009 {{.*}} WEAK   {{.*}}           weak_bar
// SYM: 00008005 {{.*}} WEAK   {{.*}} __acle_se_weak_bar
// SYM: 00020011 {{.*}} GLOBAL {{.*}}           global_baz
// SYM: 00008009 {{.*}} WEAK   {{.*}} __acle_se_global_baz
// SYM: 00020019 {{.*}} WEAK   {{.*}}           weak_qux
// SYM: 0000800d {{.*}} GLOBAL {{.*}} __acle_se_weak_qux

  .include "arm-cmse-macros.s"

  cmse_veneer global_foo, function, global, function, global
  cmse_veneer weak_bar, function, weak, function, weak
  cmse_veneer global_baz, function, global, function, weak
  cmse_veneer weak_qux, function, weak, function, global
