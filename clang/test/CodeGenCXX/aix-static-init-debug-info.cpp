// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -emit-llvm -x c++ \
// RUN:     -debug-info-kind=limited < %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -emit-llvm -x c++ \
// RUN:     -debug-info-kind=limited  < %s | \
// RUN:   FileCheck %s

struct X {
  X();
  ~X();
};

X v;

// CHECK: define internal void @__cxx_global_var_init() [[ATTR:#[0-9]+]] !dbg ![[DBGVAR16:[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN1XC1Ev(%struct.X* @v), !dbg ![[DBGVAR19:[0-9]+]]
// CHECK:   %0 = call i32 @atexit(void ()* @__dtor_v) [[ATTR:#[0-9]+]], !dbg ![[DBGVAR19b:[0-9]+]]
// CHECK:   ret void, !dbg ![[DBGVAR19]]
// CHECK: }

// CHECK: define internal void @__dtor_v() [[ATTR:#[0-9]+]] !dbg ![[DBGVAR20:[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZN1XD1Ev(%struct.X* @v), !dbg ![[DBGVAR21b:[0-9]+]]
// CHECK:   ret void, !dbg ![[DBGVAR21:[0-9]+]]
// CHECK: }

// CHECK: define internal void @__finalize_v() [[ATTR:#[0-9]+]] !dbg ![[DBGVAR22:[0-9]+]] {
// CHECK: entry:
// CHECK:   %0 = call i32 @unatexit(void ()* @__dtor_v) [[ATTR:#[0-9]+]], !dbg ![[DBGVAR24:[0-9]+]]
// CHECK:   %needs_destruct = icmp eq i32 %0, 0, !dbg ![[DBGVAR24]]
// CHECK:   br i1 %needs_destruct, label %destruct.call, label %destruct.end, !dbg ![[DBGVAR24]]

// CHECK: destruct.call:
// CHECK:   call void @__dtor_v(), !dbg ![[DBGVAR24]]
// CHECK:   br label %destruct.end, !dbg ![[DBGVAR24]]

// CHECK: destruct.end:
// CHECK:   ret void, !dbg ![[DBGVAR24]]
// CHECK: }

// CHECK: define internal void @_GLOBAL__sub_I__() [[ATTR:#[0-9]+]] !dbg ![[DBGVAR25:[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__cxx_global_var_init(), !dbg ![[DBGVAR26:[0-9]+]]
// CHECK:   ret void
// CHECK: }

// CHECK: define internal void @_GLOBAL__D_a() [[ATTR:#[0-9]+]] !dbg ![[DBGVAR27:[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @__finalize_v(), !dbg ![[DBGVAR28:[0-9]+]]
// CHECK:   ret void
// CHECK: }

// CHECK: ![[DBGVAR16]] = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, type: !{{[0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !{{[0-9]+}}, retainedNodes: !{{[0-9]+}})
// CHECK: ![[DBGVAR19]] = !DILocation(line: 14, column: 3, scope: ![[DBGVAR16]])
// CHECK: ![[DBGVAR19b]] = !DILocation(line: 0, scope: ![[DBGVAR16]])
// CHECK: ![[DBGVAR20]] = distinct !DISubprogram(name: "__dtor_v", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 14, type: !{{[0-9]+}}, scopeLine: 14, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !{{[0-9]+}}, retainedNodes: !{{[0-9]+}})
// CHECK: ![[DBGVAR21b]] = !DILocation(line: 0, scope: ![[DBGVAR20]])
// CHECK: ![[DBGVAR21]] = !DILocation(line: 14, column: 3, scope: ![[DBGVAR20]])
// CHECK: ![[DBGVAR22]] = distinct !DISubprogram(linkageName: "__finalize_v", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 14, type: !{{[0-9]+}}, scopeLine: 14, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !{{[0-9]+}}, retainedNodes: !{{[0-9]+}})
// CHECK: ![[DBGVAR24]] = !DILocation(line: 14, column: 3, scope: ![[DBGVAR22]])
// CHECK: ![[DBGVAR25]] = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I__", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, type: !{{[0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !{{[0-9]+}}, retainedNodes: !{{[0-9]+}})
// CHECK: ![[DBGVAR26]] = !DILocation(line: 0, scope: ![[DBGVAR25]])
// CHECK: ![[DBGVAR27]] = distinct !DISubprogram(linkageName: "_GLOBAL__D_a", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, type: !{{[0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !{{[0-9]+}}, retainedNodes: !{{[0-9]+}})
// CHECK: ![[DBGVAR28]] = !DILocation(line: 0, scope: ![[DBGVAR27]])
