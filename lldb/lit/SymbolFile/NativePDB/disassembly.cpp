// clang-format off
// REQUIRES: lld

// Test that we can show disassembly and source.
// RUN: %clang_cl -m64 /Z7 /GS- /GR- /c /Fo%t.obj -- %s
// RUN: lld-link /DEBUG /nodefaultlib /entry:main /OUT:%t.exe /PDB:%t.pdb -- %t.obj
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/disassembly.lldbinit | FileCheck %s

// Some context lines before
// the function.

int foo() { return 42; }

int main(int argc, char **argv) {
  foo();
  return 0;
}


// CHECK:      (lldb) disassemble --flavor=intel -m -n main
// CHECK:         13   int foo() { return 42; }
// CHECK-NEXT:    14
// CHECK-NEXT: ** 15   int main(int argc, char **argv) {
// CHECK:      disassembly.cpp.tmp.exe`main:
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+0>:  sub    rsp, 0x38
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+4>:  mov    dword ptr [rsp + 0x34], 0x0
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+12>: mov    qword ptr [rsp + 0x28], rdx
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+17>: mov    dword ptr [rsp + 0x24], ecx
// CHECK:      ** 16     foo();
// CHECK:      disassembly.cpp.tmp.exe[{{.*}}] <+21>: call   {{.*}}               ; foo at disassembly.cpp:13
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+26>: xor    ecx, ecx
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+28>: mov    dword ptr [rsp + 0x20], eax
// CHECK:      ** 17     return 0;
// CHECK-NEXT:    18   }
// CHECK-NEXT:    19
// CHECK:      disassembly.cpp.tmp.exe[{{.*}}] <+32>: mov    eax, ecx
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+34>: add    rsp, 0x38
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+38>: ret
