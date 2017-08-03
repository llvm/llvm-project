// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple thumbv7-windows -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple thumbv7-linux-gnueabihf -fms-extensions -emit-llvm -o - | FileCheck -check-prefix LINUX %s
// RUN: %clang_cc1 %s -triple i686-pc-linux -fms-extensions -emit-llvm -o - | FileCheck -check-prefix LINUX %s
// RUN: %clang_cc1 %s -triple x86_64-scei-ps4 -fms-extensions -emit-llvm -o - | FileCheck -check-prefix PS4 %s

#pragma comment(lib, "msvcrt.lib")
#pragma comment(lib, "kernel32")
#pragma comment(lib, "USER32.LIB")
#pragma comment(lib, "with space")

#define BAR "2"
#pragma comment(linker," /bar=" BAR)
#pragma comment(linker," /foo=\"foo bar\"")

// CHECK: !llvm.linker.options = !{![[msvcrt:[0-9]+]], ![[kernel32:[0-9]+]], ![[USER32:[0-9]+]], ![[space:[0-9]+]], ![[bar:[0-9]+]], ![[foo:[0-9]+]]}
// CHECK: ![[msvcrt]] = !{!"/DEFAULTLIB:msvcrt.lib"}
// CHECK: ![[kernel32]] = !{!"/DEFAULTLIB:kernel32.lib"}
// CHECK: ![[USER32]] = !{!"/DEFAULTLIB:USER32.LIB"}
// CHECK: ![[space]] = !{!"/DEFAULTLIB:\22with space.lib\22"}
// CHECK: ![[bar]] = !{!" /bar=2"}
// CHECK: ![[foo]] = !{!" /foo=\22foo bar\22"}

// LINUX: !{!"-lmsvcrt.lib"}
// LINUX: !{!"-lkernel32"}
// LINUX: !{!"-lUSER32.LIB"}
// LINUX: !{!" /bar=2"}

// PS4: !{!"\01msvcrt.lib"}
// PS4: !{!"\01kernel32"}
// PS4: !{!"\01USER32.LIB"}
// PS4: !{!"\01\22with space\22"}
