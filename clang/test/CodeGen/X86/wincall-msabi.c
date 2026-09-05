// RUN: %clang_cc1 -triple x86_64apx-unknown-linux-gnu -emit-llvm -o - %s | FileCheck --check-prefixes=APX,ALL %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck --check-prefixes=PLAIN,ALL %s

// WinCall implies the Microsoft x64 ABI, so the wincall attribute is valid on
// any x86-64 target (ELF, Mach-O, ...) and does not need to be combined with
// ms_abi. On x86_64apx, msabi alone also implies wincall; elsewhere msabi
// stays the classic MS x64 (win64cc) convention.

__attribute__((wincall)) int f_wc(int a) { return a; }
// ALL-LABEL: define dso_local x86_wincallcc i32 @f_wc

__attribute__((ms_abi, wincall)) int f_mw(int a) { return a; }
// ALL-LABEL: define dso_local x86_wincallcc i32 @f_mw

__attribute__((wincall, ms_abi)) int f_wm(int a) { return a; }
// ALL-LABEL: define dso_local x86_wincallcc i32 @f_wm

__attribute__((ms_abi)) int f_msabi(int a) { return a; }
// APX-LABEL: define dso_local x86_wincallcc i32 @f_msabi
// PLAIN-LABEL: define dso_local win64cc i32 @f_msabi
