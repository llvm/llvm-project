// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s -check-prefix=LONG-WCHAR
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -emit-llvm -o - %s | FileCheck %s -check-prefix=SHORT-WCHAR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - -fwchar-type=short -fno-signed-wchar %s | FileCheck %s -check-prefix=SHORT-WCHAR
// Note: -fno-short-wchar implies the target default is used; so there is no
// need to test this separately here.

// LONG-WCHAR: !{{[0-9]+}} = !{i32 {{[0-9]+}}, !"wchar_size", i32 4}
// SHORT-WCHAR: !{{[0-9]+}} = !{i32 {{[0-9]+}}, !"wchar_size", i32 2}
