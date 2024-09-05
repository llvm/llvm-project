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
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always -Wa,-mimplicit-it=always %s 2>&1 | FileCheck %s --check-prefix=ALWAYS
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=never -Wa,-mimplicit-it=always %s 2>&1 | FileCheck %s --check-prefix=ALWAYS
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always -Wa,-mimplicit-it=never %s 2>&1 | FileCheck %s --check-prefix=NEVER
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always -Wa,-mimplicit-it=arm %s 2>&1 | FileCheck %s --check-prefix=ARM
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always -Wa,-mimplicit-it=thumb %s 2>&1 | FileCheck %s --check-prefix=THUMB
/// Test comma separated -Wa,- arguments (latter wins).
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always,-mimplicit-it=always %s 2>&1 | FileCheck %s --check-prefix=ALWAYS
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=never,-mimplicit-it=always %s 2>&1 | FileCheck %s --check-prefix=ALWAYS
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always,-mimplicit-it=never %s 2>&1 | FileCheck %s --check-prefix=NEVER
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always,-mimplicit-it=arm %s 2>&1 | FileCheck %s --check-prefix=ARM
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always,-mimplicit-it=thumb %s 2>&1 | FileCheck %s --check-prefix=THUMB

/// Mix -implicit-it= (compiler) with -Wa,-mimplicit-it= (assembler), the
/// last one set takes priority.
// RUN: %clang -target arm-linux-gnueabi -### -mimplicit-it=always -Wa,-mimplicit-it=always %S/Inputs/wildcard1.c 2>&1 | FileCheck %s --check-prefix=ALWAYS
// RUN: %clang -target arm-linux-gnueabi -### -mimplicit-it=never -Wa,-mimplicit-it=always %S/Inputs/wildcard1.c 2>&1 | FileCheck %s --check-prefix=ALWAYS
// RUN: %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=never -mimplicit-it=always %S/Inputs/wildcard1.c 2>&1 | FileCheck %s --check-prefix=ALWAYS

/// Test invalid input.
// RUN: not %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=foo %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %clang -target arm-linux-gnueabi -### -Xassembler -mimplicit-it=foo %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always -Wa,-mimplicit-it=foo %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %clang -target arm-linux-gnueabi -### -Wa,-mimplicit-it=always,-mimplicit-it=foo %s 2>&1 | FileCheck %s --check-prefix=INVALID


/// Check that the argument we ignore is still marked as used.
// ALWAYS-NOT: warning: argument unused during compilation: {{.*}}-mimplicit-it={{.*}}
/// Check that there isn't a second -arm-implicit-it before or after the one
/// that was the indended match.
// ALWAYS-NOT: "-arm-implicit-it={{.*}}"
// ALWAYS: "-mllvm" "-arm-implicit-it=always"
// ALWAYS-NOT: "-arm-implicit-it={{.*}}"
// NEVER-NOT: "-arm-implicit-it={{.*}}"
// NEVER: "-mllvm" "-arm-implicit-it=never"
// NEVER-NOT: "-arm-implicit-it={{.*}}"
// ARM: "-mllvm" "-arm-implicit-it=arm"
// THUMB: "-mllvm" "-arm-implicit-it=thumb"
// INVALID: error: unsupported argument 'foo' to option '-Wa,-mimplicit-it='
