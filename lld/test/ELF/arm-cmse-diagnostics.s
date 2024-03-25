// REQUIRES: arm
/// Test CMSE diagnostics.

// RUN: rm -rf %t && split-file %s %t && cd %t

/// Test diagnostics emitted during checks of the CMSE import library
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.base lib -o lib.o
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.base app -I %S/Inputs -o app.o
// RUN: llvm-objcopy --redefine-sym=entry7_duplicate=entry6_duplicate lib.o
// RUN: not ld.lld --cmse-implib --in-implib=lib.o app.o -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ERR_IMPLIB
// RUN: not ld.lld --cmse-implib --in-implib=lib.o --in-implib=lib.o app.o -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ERR_MULT_INIMPLIB
// RUN: not ld.lld --in-implib=lib.o app.o -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ERR_IN_IMPLIB
// RUN: not ld.lld --out-implib=out.lib app.o -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ERR_OUT_IMPLIB
// RUN: not ld.lld --out-implib=out.lib --in-implib=lib.o app.o -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ERR_IN_IMPLIB,ERR_OUT_IMPLIB

// ERR_IMPLIB: error: CMSE symbol 'entry_not_external' in import library '{{.*}}' is not global
// ERR_IMPLIB: error: CMSE symbol 'entry_not_absolute' in import library '{{.*}}' is not absolute
// ERR_IMPLIB: error: CMSE symbol 'entry_not_function' in import library '{{.*}}' is not a Thumb function definition
// ERR_IMPLIB: error: CMSE symbol 'entry_not_thumb' in import library '{{.*}}' is not a Thumb function definition
// ERR_IMPLIB: warning: CMSE symbol 'entry5_incorrect_size' in import library '{{.*}}' does not have correct size of 8 bytes
// ERR_IMPLIB: error: CMSE symbol 'entry6_duplicate' is multiply defined in import library '{{.*}}'
// ERR_MULT_INIMPLIB: error: multiple CMSE import libraries not supported
// ERR_IN_IMPLIB: error: --in-implib may not be used without --cmse-implib
// ERR_OUT_IMPLIB: error: --out-implib may not be used without --cmse-implib

/// CMSE Only supported by ARMv8-M architecture or later.
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv8m.base app -I %S/Inputs -o app1.o
// RUN: ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 -o app1 app1.o 2>&1 | FileCheck /dev/null --implicit-check-not=error:

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv8m.main app -I %S/Inputs -o app2.o
// RUN: ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 -o app2 app2.o 2>&1 | FileCheck /dev/null --implicit-check-not=error:

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv8.1m.main app -I %S/Inputs -o app3.o
// RUN: ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 -o app3 app3.o 2>&1 | FileCheck /dev/null --implicit-check-not=error:

/// Expect errors for other architectures.
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv9a app -I %S/Inputs -o app4.o
// RUN: not ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 -o app4 app4.o 2>&1 | FileCheck %s --check-prefixes=ERR_ARCH

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7-m app -I %S/Inputs -o app5.o
// RUN: not ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 -o app5 app5.o 2>&1 | FileCheck %s --check-prefixes=ERR_ARCH

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv8-m app -I %S/Inputs -o app6.o
// RUN: not ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 -o app6 app6.o 2>&1 | FileCheck %s --check-prefixes=ERR_ARCH

/// Invalid triple defaults to v4T. Error.
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumb app -I %S/Inputs -o app7.o
// RUN: not ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 -o app7 app7.o  2>&1 | FileCheck %s --check-prefixes=ERR_ARCH

/// No build attributes. Error.
// RUN: llvm-mc -filetype=obj -triple=thumb app -I %S/Inputs -o app8.o
// RUN: not ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 -o app8 app8.o  2>&1 | FileCheck %s --check-prefixes=ERR_ARCH

// ERR_ARCH: CMSE is only supported by ARMv8-M architecture or later

/// Test that the linker diagnoses cases where the linker synthesized veneer addresses
/// specified by the CMSE input library cannot be placed at the .gnu.sgstubs section address.

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.main app -I %S/Inputs -o 1.o
/// Create a CMSE import library with a secure gateway veneer at 0x10000
// RUN: ld.lld --cmse-implib --section-start .gnu.sgstubs=0x10000 1.o -o 1 --out-implib=1.lib 2>&1 | FileCheck /dev/null --implicit-check-not=error:
/// Create a new import library with the secure gateway veneer and .gnu.sgstubs specified at the same address
// RUN: ld.lld --cmse-implib --section-start .gnu.sgstubs=0x10000 1.o -o 2 --out-implib=2.lib --in-implib=1.lib 2>&1 | FileCheck /dev/null --implicit-check-not=error:
/// Create a new import library with the secure gateway veneer specified at a same address but .gnu.sgstubs at a higher address.
// RUN: not ld.lld --cmse-implib --section-start .gnu.sgstubs=0x11000 1.o -o 3 --out-implib=3.lib --in-implib=1.lib 2>&1 | FileCheck %s --check-prefixes=ERR_ADDR
/// Create a new import library with the secure gateway veneer specified at a same address but .gnu.sgstubs at a lower address.
// RUN: not ld.lld --cmse-implib --section-start .gnu.sgstubs=0x9000 1.o -o 4 --out-implib=4.lib --in-implib=1.lib 2>&1 | FileCheck %s --check-prefixes=ERR_ADDR

// ERR_ADDR: error: start address of '.gnu.sgstubs' is different from previous link

/// Test that the address of .gnu.sgstubs can be specified via command line or linker script.
/// Test that the linker errors when the address of .gnu.sgstubs is not specified using either method.

// RUN: ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 --script with_sgstubs.script 1.o -o 1 2>&1 | FileCheck /dev/null --implicit-check-not=error:
// RUN: ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 --script wout_sgstubs.script 1.o -o 2 2>&1 | FileCheck /dev/null --implicit-check-not=error:
// RUN: ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 1.o -o 3 2>&1 | FileCheck /dev/null --implicit-check-not=error:
// RUN: ld.lld --cmse-implib -Ttext=0x8000 --script with_sgstubs.script 1.o -o 4 2>&1 | FileCheck /dev/null --implicit-check-not=error:
// RUN: not ld.lld --cmse-implib -Ttext=0x8000 --script wout_sgstubs.script 1.o -o 5 2>&1 | FileCheck %s --check-prefixes=ERR_NOADDR

// RUN: llvm-readelf -S 1 | FileCheck %s --check-prefixes=ADDRCMDLINE
// RUN: llvm-readelf -S 2 | FileCheck %s --check-prefixes=ADDRCMDLINE
// RUN: llvm-readelf -S 3 | FileCheck %s --check-prefixes=ADDRCMDLINE
// RUN: llvm-readelf -S 4 | FileCheck %s --check-prefixes=ADDRLNKSCRIPT

// ERR_NOADDR: error: no address assigned to the veneers output section .gnu.sgstubs

///                       Name          Type         Address    Off   Size ES Flg Lk Inf Al
// ADDRCMDLINE:   .gnu.sgstubs      PROGBITS        00020000 020000 000008 08  AX  0   0 32
// ADDRLNKSCRIPT: .gnu.sgstubs      PROGBITS        00040000 040000 000008 08  AX  0   0 32

/// Test diagnostics emitted during symbol attribute checks.

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -I %S/Inputs --triple=thumbv8m.base symattr -o symattr.o
// RUN: not ld.lld --cmse-implib symattr.o -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ERR_SYMATTR

// ERR_SYMATTR-NOT: __acle_se_valid_{{.*}}
// ERR_SYMATTR: error: {{.*}}: cmse special symbol '__acle_se_invalid_1' is not a Thumb function definition
// ERR_SYMATTR: error: {{.*}}: cmse special symbol '__acle_se_invalid_2' is not a Thumb function definition
// ERR_SYMATTR: error: {{.*}}: cmse entry symbol 'invalid_3' is not a Thumb function definition
// ERR_SYMATTR: error: {{.*}}: cmse entry symbol 'invalid_4' is not a Thumb function definition
// ERR_SYMATTR: error: {{.*}}: cmse special symbol '__acle_se_invalid_5' detected, but no associated entry function definition 'invalid_5' with external linkage found
// ERR_SYMATTR: error: {{.*}}: cmse special symbol '__acle_se_invalid_6' detected, but no associated entry function definition 'invalid_6' with external linkage found
// ERR_SYMATTR: error: {{.*}}: cmse special symbol '__acle_se_invalid_7' is not a Thumb function definition
// ERR_SYMATTR: error: {{.*}}: cmse special symbol '__acle_se_invalid_8' detected, but no associated entry function definition 'invalid_8' with external linkage found
// ERR_SYMATTR: error: {{.*}}: cmse special symbol '__acle_se_invalid_9' cannot be an absolute symbol
// ERR_SYMATTR: error: {{.*}}: cmse special symbol '__acle_se_invalid_10' cannot be an absolute symbol
// ERR_SYMATTR: error: {{.*}}: cmse special symbol '__acle_se_invalid_11' is not a Thumb function definition
// ERR_SYMATTR: error: {{.*}}: cmse entry symbol 'invalid_12' is not a Thumb function definition

/// Test diagnostics emitted when a symbol is removed from a later version of the import library.
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -I %S/Inputs --triple=thumbv8m.base libv1 -o libv1.o
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -I %S/Inputs --triple=thumbv8m.base libv2 -o libv2.o
// RUN: ld.lld -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 --cmse-implib libv1.o --out-implib=libv1.lib -o /dev/null
// RUN: ld.lld -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 --cmse-implib libv2.o --in-implib=libv1.lib --out-implib=libv2.lib -o /dev/null 2>&1 | FileCheck %s --check-prefixes=WARN_MISSING
// RUN: ld.lld -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 --cmse-implib libv1.o --in-implib=libv2.lib -o /dev/null 2>&1 | FileCheck %s --check-prefixes=WARN_NEWENTRY

// WARN_MISSING: warning: entry function 'bar' from CMSE import library is not present in secure application
// WARN_NEWENTRY: warning: new entry function 'bar' introduced but no output import library specified

//--- with_sgstubs.script
SECTIONS {
  .text : { *(.text) }
  .gnu.sgstubs 0x40000 : { *(.gnu.sgstubs*) }
}

//--- wout_sgstubs.script
SECTIONS {
  .text : { *(.text) }
}

//--- app
  .include "arm-cmse-macros.s"
  .text
  .thumb

cmse_veneer entry, function, global, function, global

//--- lib
    .text
    .thumb

/// Symbol not absolute.
    .global entry_not_absolute
    .type entry_not_absolute, STT_FUNC
    .thumb_func
entry_not_absolute:
    .size entry_not_absolute, 8

/// Symbol not global or weak.
    .local entry_not_external
    .type entry_not_external, STT_FUNC
entry_not_external=0x1001
    .size entry_not_external, 8

/// Symbol not the function type.
    .global entry_not_function
    .type entry_not_function, STT_NOTYPE
entry_not_function=0x1001
    .size entry_not_function, 8

/// Symbol not a Thumb code symbol.
    .global entry_not_thumb
    .type entry_not_thumb, STT_FUNC
entry_not_thumb=0x1000
    .size entry_not_thumb, 8

/// Symbol with incorrect size.
    .global entry5_incorrect_size
    .type entry5_incorrect_size, STT_FUNC
entry5_incorrect_size=0x1009
    .size entry5_incorrect_size, 6

/// Duplicate symbols.
    .global entry6_duplicate
    .type entry6_duplicate, STT_FUNC
entry6_duplicate=0x1001
    .size entry6_duplicate, 8

/// entry7_duplicate gets renamed to entry6_duplicate by llvm-objcopy.
    .global entry7_duplicate
    .type entry7_duplicate, STT_FUNC
entry7_duplicate=0x1009
    .size entry7_duplicate, 8

//--- symattr
.include "arm-cmse-macros.s"

  .text
  .thumb

/// Valid sequences
/// both sym and __acle_se_sym should be global or weak Thumb code symbols.
  cmse_veneer valid_1, function, global, function, global
  cmse_veneer valid_2, function,   weak, function,   weak
  cmse_veneer valid_3, function,   weak, function, global
  cmse_veneer valid_4, function, global, function,   weak

/// Invalid sequences
/// __acle_se_sym is an object
  cmse_veneer invalid_1, function, global,   object, global
  cmse_veneer invalid_2, function, global,   object,   weak
/// sym is an object
  cmse_veneer invalid_3,   object, global, function, global
  cmse_veneer invalid_4,   object, global, function,   weak
/// sym is local
  cmse_veneer invalid_5, function,  local, function, global
  cmse_veneer invalid_6, function,  local, function,   weak

/// __acle_se_invalid_7 not defined.
  .global invalid_7
	.type	invalid_7, %function
  .global __acle_se_invalid_7
  .thumb_func
invalid_7:

/// invalid_8 not defined.
  .global __acle_se_invalid_8
  .thumb_func
__acle_se_invalid_8:

// Absolute symbols with same values
  .global invalid_9
  .global __acle_se_invalid_9
	.type	__acle_se_invalid_9, %function
	.type	invalid_9, %function
__acle_se_invalid_9=0x1001
invalid_9=0x1001
	.size	invalid_9, 0
  .size __acle_se_invalid_9, 0

// Absolute symbols with different values
	.align 2
	.global	__acle_se_invalid_10
	.global	invalid_10
	.type	__acle_se_invalid_10, %function
	.type	invalid_10, %function
__acle_se_invalid_10 = 0x10001
invalid_10 = 0x10005
	.size	invalid_10, 0
  .size __acle_se_invalid_10, 0

  .section nonthumb
  .thumb
  .align  2
  .global  invalid_11
  .global  __acle_se_invalid_11
  .type  invalid_11, %function
  .type  __acle_se_invalid_11, %function
invalid_11:
  .size  invalid_11, .-invalid_11
/// Invalid non-thumb function symbol __acle_se_invalid_11
__acle_se_invalid_11=0x1000

  .global  invalid_12
  .global  __acle_se_invalid_12
  .type  invalid_12, %function
  .type  __acle_se_invalid_12, %function
/// Invalid non-thumb function symbol invalid_12
invalid_12=0x1000
  .thumb
__acle_se_invalid_12:
  .size  __acle_se_invalid_12, .-__acle_se_invalid_12

//--- libv1
.include "arm-cmse-macros.s"

  .text
  .thumb

/// Import library version 1 with foo and bar
  cmse_veneer foo, function, global, function, global
  cmse_veneer bar, function,   weak, function,   weak

//--- libv2
.include "arm-cmse-macros.s"

  .text
  .thumb

/// Import library version 2 with bar missing.
  cmse_veneer foo, function, global, function, global
