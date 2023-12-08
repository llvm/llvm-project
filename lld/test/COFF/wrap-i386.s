// REQUIRES: x86
// RUN: split-file %s %t.dir
// RUN: llvm-mc -filetype=obj -triple=i686-win32-gnu %t.dir/main.s -o %t.main.obj
// RUN: llvm-mc -filetype=obj -triple=i686-win32-gnu %t.dir/other.s -o %t.other.obj

// RUN: lld-link -out:%t.exe %t.main.obj %t.other.obj -entry:entry -subsystem:console -debug:symtab -safeseh:no -wrap:foo -wrap:nosuchsym
// RUN: llvm-objdump -d --print-imm-hex %t.exe | FileCheck %s

// CHECK: <_entry>:
// CHECK-NEXT: movl $0x11010, %edx
// CHECK-NEXT: movl $0x11010, %edx
// CHECK-NEXT: movl $0x11000, %edx

// RUN: llvm-readobj --symbols %t.exe > %t.dump
// RUN: FileCheck --check-prefix=SYM1 %s < %t.dump
// RUN: FileCheck --check-prefix=SYM2 %s < %t.dump
// RUN: FileCheck --check-prefix=SYM3 %s < %t.dump

// _foo = 0x00011000 = 69632
// ___wrap_foo = 0x00011010 = 69648
// SYM1:      Name: _foo
// SYM1-NEXT: Value: 69632
// SYM1-NEXT: Section: IMAGE_SYM_ABSOLUTE
// SYM1-NEXT: BaseType: Null
// SYM1-NEXT: ComplexType: Null
// SYM1-NEXT: StorageClass: External
// SYM2:      Name: ___wrap_foo
// SYM2-NEXT: Value: 69648
// SYM2-NEXT: Section: IMAGE_SYM_ABSOLUTE
// SYM2-NEXT: BaseType: Null
// SYM2-NEXT: ComplexType: Null
// SYM2-NEXT: StorageClass: External
// SYM3-NOT:  Name: ___real_foo

#--- main.s
.global _entry
_entry:
  movl $_foo, %edx
  movl $___wrap_foo, %edx
  movl $___real_foo, %edx

#--- other.s
.global _foo
.global ___wrap_foo
.global ___real_foo

_foo = 0x11000
___wrap_foo = 0x11010
___real_foo = 0x11020
