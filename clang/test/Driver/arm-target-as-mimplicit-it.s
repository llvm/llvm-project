/// Simple tests for valid input.
/// -Wa,-implicit-it=
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always %s 2>&1 | FileCheck %s --check-prefix=ALWAYS
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=never %s 2>&1 | FileCheck %s --check-prefix=NEVER
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=arm %s 2>&1 | FileCheck %s --check-prefix=ARM
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=thumb %s 2>&1 | FileCheck %s --check-prefix=THUMB
/// -Xassembler -mimplicit-it=
// RUN: %clang -target arm-linux-gnueabi -### -Xassembler -mimplicit-it=always %s 2>&1 | FileCheck %s --check-prefix=ALWAYS
// RUN: %clang -target arm-linux-gnueabi -### -Xassembler -mimplicit-it=never %s 2>&1 | FileCheck %s --check-prefix=NEVER
// RUN: %clang -target arm-linux-gnueabi -### -Xassembler -mimplicit-it=arm %s 2>&1 | FileCheck %s --check-prefix=ARM
// RUN: %clang -target arm-linux-gnueabi -### -Xassembler -mimplicit-it=thumb %s 2>&1 | FileCheck %s --check-prefix=THUMB
/// Test space separated -Wa,- arguments (latter wins).
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=never -Wa,-mimplicit-it=always %s 2>&1 | FileCheck %s --check-prefix=ALWAYS
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always -Wa,-mimplicit-it=never %s 2>&1 | FileCheck %s --check-prefix=NEVER
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always -Wa,-mimplicit-it=arm %s 2>&1 | FileCheck %s --check-prefix=ARM
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always -Wa,-mimplicit-it=thumb %s 2>&1 | FileCheck %s --check-prefix=THUMB
/// Test comma separated -Wa,- arguments (latter wins).
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=never,-mimplicit-it=always %s 2>&1 | FileCheck %s --check-prefix=ALWAYS
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always,-mimplicit-it=never %s 2>&1 | FileCheck %s --check-prefix=NEVER
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always,-mimplicit-it=arm %s 2>&1 | FileCheck %s --check-prefix=ARM
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always,-mimplicit-it=thumb %s 2>&1 | FileCheck %s --check-prefix=THUMB

/// Mix -implicit-it= (compiler) with -Wa,-mimplicit-it= (assembler), assembler
/// takes priority. -mllvm -arm-implicit-it= will be repeated, with the
/// assembler flag appearing last (latter wins).
// RUN: %clang -target arm-linux-gnueabi -### -mimplicit-it=never -Wa,-mimplicit-it=always %S/Inputs/wildcard1.c 2>&1 | FileCheck %s --check-prefix=NEVER_ALWAYS
// RUN: %clang -target arm-linux-gnueabi -### -mimplicit-it=always -Wa,-mimplicit-it=never %S/Inputs/wildcard1.c 2>&1 | FileCheck %s --check-prefix=ALWAYS_NEVER
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=never -mimplicit-it=always %S/Inputs/wildcard1.c 2>&1 | FileCheck %s --check-prefix=ALWAYS_NEVER

/// Test invalid input.
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=foo %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: %clang -target arm-linux-gnueabi -### -Xassembler -mimplicit-it=foo %s 2>&1 | FileCheck %s --check-prefix=XINVALID
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always -Wa,-mimplicit-it=foo %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always,-mimplicit-it=foo %s 2>&1 | FileCheck %s --check-prefix=INVALID


// ALWAYS: "-mllvm" "-arm-implicit-it=always"
// NEVER: "-mllvm" "-arm-implicit-it=never"
// ARM: "-mllvm" "-arm-implicit-it=arm"
// THUMB: "-mllvm" "-arm-implicit-it=thumb"
// NEVER_ALWAYS: "-mllvm" "-arm-implicit-it=never" "-mllvm" "-arm-implicit-it=always"
// ALWAYS_NEVER: "-mllvm" "-arm-implicit-it=always" "-mllvm" "-arm-implicit-it=never"
// INVALID: error: unsupported argument '-mimplicit-it=foo' to option 'Wa,'
// XINVALID: error: unsupported argument '-mimplicit-it=foo' to option 'Xassembler'
