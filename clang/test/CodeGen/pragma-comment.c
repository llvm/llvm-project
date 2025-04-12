// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple thumbv7-windows -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple thumbv7-linux-gnueabihf -fms-extensions -emit-llvm -o - | FileCheck -check-prefix ELF %s --implicit-check-not llvm.linker.options
// RUN: %clang_cc1 %s -triple i686-pc-linux -fms-extensions -emit-llvm -o - | FileCheck -check-prefix ELF %s --implicit-check-not llvm.linker.options
// RUN: %clang_cc1 %s -triple x86_64-scei-ps4 -fms-extensions -emit-llvm -o - | FileCheck -check-prefix ELF %s --implicit-check-not llvm.linker.options
// RUN: %clang_cc1 %s -triple x86_64-sie-ps5 -fms-extensions -emit-llvm -o - | FileCheck -check-prefix ELF %s --implicit-check-not llvm.linker.options
// RUN: %clang_cc1 %s -triple aarch64-windows-msvc -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple mipsel-windows-msvc -fms-extensions -emit-llvm -o - | FileCheck %s

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

// ELF: !llvm.dependent-libraries = !{![[msvcrt:[0-9]+]], ![[kernel32:[0-9]+]], ![[USER32:[0-9]+]], ![[space:[0-9]+]]
// ELF: ![[msvcrt]] = !{!"msvcrt.lib"}
// ELF: ![[kernel32]] = !{!"kernel32"}
// ELF: ![[USER32]] = !{!"USER32.LIB"}
// ELF: ![[space]] = !{!"with space"}
// ELF-NOT: bar
// ELF-NOT: foo
