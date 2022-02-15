// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=address -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fsanitize=address -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=WINDOWS
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fsanitize=address -fwritable-strings -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=WINWRITE

// On Linux (and basically every non-MS target) string literals are emitted with
// private linkage, which means ASan can freely instrument them. On Windows,
// they are emitted with comdats. ASan's global instrumentation code for COFF
// knows how to make the metadata comdat associative, so the string literal
// global is only registered if the instrumented global prevails during linking.

const char *foo(void) { return "asdf"; }

// LINUX: @.str = private unnamed_addr constant [5 x i8] c"asdf\00", align 1

// WINDOWS: @"??_C@_04JIHMPGLA@asdf?$AA@" = linkonce_odr dso_local unnamed_addr constant [5 x i8] c"asdf\00", comdat, align 1

// WINWRITE: @.str = private unnamed_addr global [5 x i8] c"asdf\00", align 1
