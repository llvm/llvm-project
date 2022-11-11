// RUN: %clang_cc1 %s -triple i386-pc-windows-msvc19.0.0 -emit-llvm \
// RUN:     -debug-info-kind=line-tables-only -fms-extensions -o - | FileCheck %s
class __declspec(dllexport) A {
  A(int * = new int) {}
};
// CHECK: define {{.*}}void @"??_FA@@AAEXXZ"
// CHECK-SAME: !dbg ![[SP:[0-9]+]]
// CHECK-NOT: {{ret }}
// CHECK: call x86_thiscallcc noundef ptr @"??0A@@AAE@PAH@Z"
// CHECK-SAME: !dbg ![[DBG:[0-9]+]]
// CHECK: ret void, !dbg
//
// CHECK: ![[SP]] = distinct !DISubprogram(
// CHECK-SAME:          line: 4
// CHECK-SAME:          DIFlagArtificial
// CHECK-SAME:          DISPFlagDefinition
// CHECK-SAME:          ){{$}}
//
// CHECK: ![[DBG]] = !DILocation(line: 0
