//// Explicitly enabled:
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone   \
// RUN:     -emit-llvm  -fexperimental-assignment-tracking=enabled %s -o - -O1 \
// RUN: | FileCheck %s --check-prefixes=ENABLE

//// Explicitly disabled:
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone   \
// RUN:     -emit-llvm  %s -o - -fexperimental-assignment-tracking=disabled -O1\
// RUN: | FileCheck %s --check-prefixes=DISABLE

//// Enabled by default:
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone   \
// RUN:     -emit-llvm  %s -o - -O1                                            \
// RUN: | FileCheck %s --check-prefixes=ENABLE

//// Disabled at O0 unless forced.
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone   \
// RUN:     -emit-llvm %s -o - -fexperimental-assignment-tracking=enabled      \
// RUN:     -O0 -disable-O0-optnone                                           \
// RUN: | FileCheck %s --check-prefixes=DISABLE
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone   \
// RUN:     -emit-llvm %s -o - -fexperimental-assignment-tracking=forced       \
// RUN:     -O0 -disable-O0-optnone                                            \
// RUN: | FileCheck %s --check-prefixes=ENABLE

//// Disabled for LTO and thinLTO unless forced.
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone   \
// RUN:     -emit-llvm %s -o - -fexperimental-assignment-tracking=enabled      \
// RUN:     -O1 -flto=full                                                     \
// RUN: | FileCheck %s --check-prefixes=DISABLE
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone   \
// RUN:     -emit-llvm %s -o - -fexperimental-assignment-tracking=enabled      \
// RUN:     -O1 -flto=thin                                                     \
// RUN: | FileCheck %s --check-prefixes=DISABLE
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone   \
// RUN:     -emit-llvm %s -o - -fexperimental-assignment-tracking=forced       \
// RUN:     -O1 -flto=full                                                     \
// RUN: | FileCheck %s --check-prefixes=ENABLE
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone   \
// RUN:     -emit-llvm %s -o - -fexperimental-assignment-tracking=forced       \
// RUN:     -O1 -flto=thin                                                     \
// RUN: | FileCheck %s --check-prefixes=ENABLE

//// Disabled for LLDB debugger tuning unless forced.
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone   \
// RUN:     -emit-llvm %s -o - -fexperimental-assignment-tracking=enabled      \
// RUN:     -O1 -debugger-tuning=lldb                                          \
// RUN: | FileCheck %s --check-prefixes=DISABLE
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone   \
// RUN:     -emit-llvm %s -o - -fexperimental-assignment-tracking=forced       \
// RUN:     -O1 -debugger-tuning=lldb                                          \
// RUN: | FileCheck %s --check-prefixes=ENABLE

// Check the assignment-tracking module flag appears in the output when the
// flag -fexperimental-assignment-tracking is set to 'enabled' (in some cases)
// or 'forced' (always), and is does not appear when the flag is set to
// 'disabled' (default).

// ENABLE: "debug-info-assignment-tracking"
// DISABLE-NOT: "debug-info-assignment-tracking"
//// Check there's actually any output at all.
// DISABLE: llvm.module.flags

void fun(int a) {}
