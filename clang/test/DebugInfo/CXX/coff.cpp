// REQUIRES: x86-registered-target

// Check that CodeView compiler version is emitted even when debug info is otherwise disabled.

// RUN: %clang --target=i686-pc-windows-msvc -S -emit-llvm %s -o - | FileCheck --check-prefix=IR %s
// IR: !llvm.dbg.cu = !{!0}
// IR: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "{{.*}}", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)

// RUN: %clang --target=i686-pc-windows-msvc -c %s -o %t.o
// RUN: llvm-readobj --codeview %t.o | FileCheck %s
// CHECK:      CodeViewDebugInfo [
// CHECK-NEXT:   Section: .debug$S (4)
// CHECK-NEXT:   Magic: 0x4
// CHECK-NEXT:   Subsection [
// CHECK-NEXT:     SubSectionType: Symbols (0xF1)
// CHECK-NEXT:     SubSectionSize:
// CHECK-NEXT:     ObjNameSym {
// CHECK-NEXT:       Kind: S_OBJNAME (0x1101)
// CHECK-NEXT:       Signature: 0x0
// CHECK-NEXT:       ObjectName:
// CHECK-NEXT:     }
// CHECK-NEXT:     Compile3Sym {
// CHECK-NEXT:       Kind: S_COMPILE3 (0x113C)
// CHECK-NEXT:       Language: Cpp (0x1)
// CHECK-NEXT:       Flags [ (0x0)
// CHECK-NEXT:       ]
// CHECK-NEXT:       Machine: Pentium3 (0x7)
// CHECK-NEXT:       FrontendVersion:
// CHECK-NEXT:       BackendVersion:
// CHECK-NEXT:       VersionName: {{.*}}clang version
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: ]

int main() {
  return 0;
}
