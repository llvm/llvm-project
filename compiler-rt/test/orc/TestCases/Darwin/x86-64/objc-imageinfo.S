// Test merging of __objc_imageinfo flags and ensure we can run mixed objc and
// swift code in a single jit dylib.

// REQUIRES: system-darwin && asserts

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: (cd %t; %clang -c *.S)

// Check individual versions are loadable.

// RUN: %llvm_jitlink -debug-only=orc %t/main.o %t/objc_old.o 2>&1 | FileCheck %s -check-prefix=OLD
// OLD: MachOPlatform: Registered __objc_imageinfo for main
// OLD-SAME: flags = 0x0000
// RUN: %llvm_jitlink -debug-only=orc %t/main.o %t/objc_new.o 2>&1 | FileCheck %s -check-prefix=NEW
// NEW: MachOPlatform: Registered __objc_imageinfo for main
// NEW-SAME: flags = 0x0040
// RUN: %llvm_jitlink -debug-only=orc %t/main.o %t/swift_4.o 2>&1 | FileCheck %s -check-prefix=SWIFT_4
// SWIFT_4: MachOPlatform: Registered __objc_imageinfo for main
// SWIFT_4-SAME: flags = 0x0640
// RUN: %llvm_jitlink -debug-only=orc %t/main.o %t/swift_5.o 2>&1 | FileCheck %s -check-prefix=SWIFT_5
// SWIFT_5: MachOPlatform: Registered __objc_imageinfo for main
// SWIFT_5-SAME: flags = 0x5000740
// RUN: %llvm_jitlink -debug-only=orc %t/main.o %t/swift_59.o 2>&1 | FileCheck %s -check-prefix=SWIFT_59
// SWIFT_59: MachOPlatform: Registered __objc_imageinfo for main
// SWIFT_59-SAME: flags = 0x5090740

// Check error conditions.

// RUN: not %llvm_jitlink %t/main.o %t/swift_4.o %t/swift_5.o 2>&1 | FileCheck %s -check-prefix=SWIFT_ABI
// SWIFT_ABI: Swift ABI version in {{.*}} does not match first registered flags

// Check merging.

// Take the lowest swift version.
// RUN: %llvm_jitlink -debug-only=orc %t/main.o %t/swift_59.o %t/swift_5.o 2>&1 | FileCheck %s -check-prefix=SWIFT_MIX1
// SWIFT_MIX1: MachOPlatform: Merging __objc_imageinfo flags for main {{.*}} -> 0x5000740

// Add swift to objc.
// RUN: %llvm_jitlink -debug-only=orc %t/main.o %t/swift_59.o %t/objc_new.o 2>&1 | FileCheck %s -check-prefix=SWIFT_MIX2
// SWIFT_MIX2: MachOPlatform: Merging __objc_imageinfo flags for main {{.*}} -> 0x5090740

// Add multiple swift to objc.
// RUN: %llvm_jitlink -debug-only=orc %t/main.o %t/swift_59.o %t/swift_5.o %t/objc_new.o 2>&1 | FileCheck %s -check-prefix=SWIFT_MIX3
// SWIFT_MIX3: MachOPlatform: Merging __objc_imageinfo flags for main {{.*}} -> 0x5000740

// Disable categories.
// RUN: %llvm_jitlink -debug-only=orc %t/main.o %t/objc_old.o %t/objc_new.o 2>&1 | FileCheck %s -check-prefix=SWIFT_MIX4
// SWIFT_MIX4: MachOPlatform: Merging __objc_imageinfo flags for main {{.*}} -> 0x0000

// Disable signed class_ro.
// RUN: %llvm_jitlink -debug-only=orc %t/main.o %t/objc_new.o %t/objc_new_signed_ro.o 2>&1 | FileCheck %s -check-prefix=SWIFT_MIX5
// SWIFT_MIX5: MachOPlatform: Merging __objc_imageinfo flags for main {{.*}} -> 0x0040

//--- main.S
.section  __TEXT,__text,regular,pure_instructions
.globl _main
_main:
  xorl %eax, %eax
  ret

//--- objc_old.S
.section  __TEXT,__text,regular,pure_instructions
.globl _objc1
_objc1:
  ret

  .section  __DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
  .long 0
  .long 0

//--- objc_new.S
.section  __TEXT,__text,regular,pure_instructions
.globl _objc2
_objc2:
  ret

  .section  __DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
  .long 0
  .long 64

//--- objc_new_signed_ro.S
.section  __TEXT,__text,regular,pure_instructions
.globl _objc3
_objc3:
  ret

  .section  __DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
  .long 0
  .long 80

//--- swift_4.S
.section  __TEXT,__text,regular,pure_instructions
.globl _swift4
_swift4:
  ret

  .section  __DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
  .long 0
  .long 1600

//--- swift_5.S
.section  __TEXT,__text,regular,pure_instructions
.globl _swift5
_swift5:
  ret

  .section  __DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
  .long 0
  .long 83887936

//--- swift_59.S
.section  __TEXT,__text,regular,pure_instructions
.globl _swift59
_swift59:
  ret

  .section  __DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
  .long 0
  .long 84477760

