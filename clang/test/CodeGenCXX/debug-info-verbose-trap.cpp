// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++20 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK-LABEL: define void @_Z2f0v()
// CHECK: call void @llvm.trap(), !dbg ![[LOC17:.*]]

// CHECK-LABEL: define void @_Z2f1v()
// CHECK: call void @llvm.trap(), !dbg ![[LOC23:.*]]
// CHECK: call void @llvm.trap(), !dbg ![[LOC25:.*]]

// CHECK-LABEL: define void @_Z2f3v()
// CHECK: call void @_Z2f2IXadsoKcL_ZL8constMsgEEEEvv()

// CHECK-LABEL: define internal void @_Z2f2IXadsoKcL_ZL8constMsgEEEEvv()
// CHECK: call void @llvm.trap(), !dbg ![[LOC36:.*]]

// CHECK: ![[FILESCOPE:.*]] = !DIFile(filename:
// CHECK: ![[SUBPROG14:.*]] = distinct !DISubprogram(name: "f0", linkageName: "_Z2f0v",
// CHECK: ![[LOC17]] = !DILocation(line: 0, scope: ![[SUBPROG18:.*]], inlinedAt: ![[LOC20:.*]])
// CHECK: ![[SUBPROG18]] = distinct !DISubprogram(name: "__llvm_verbose_trap: Argument_must_not_be_null", scope: ![[FILESCOPE]], file: ![[FILESCOPE]], type: !{{.*}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !{{.*}})
// CHECK: ![[LOC20]] = !DILocation(line: 34, column: 3, scope: ![[SUBPROG14]])
// CHECK: ![[SUBPROG22:.*]] = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v",
// CHECK: ![[LOC23]] = !DILocation(line: 0, scope: ![[SUBPROG18]], inlinedAt: ![[LOC24:.*]])
// CHECK: ![[LOC24]] = !DILocation(line: 38, column: 3, scope: ![[SUBPROG22]])
// CHECK: ![[LOC25]] = !DILocation(line: 0, scope: ![[SUBPROG26:.*]], inlinedAt: ![[LOC27:.*]])
// CHECK: ![[SUBPROG26]] = distinct !DISubprogram(name: "__llvm_verbose_trap: hello", scope: ![[FILESCOPE]], file: ![[FILESCOPE]], type: !{{.*}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !{{.*}})
// CHECK: ![[LOC27]] = !DILocation(line: 39, column: 3, scope: ![[SUBPROG22]])
// CHECK: ![[SUBPROG32:.*]] = distinct !DISubprogram(name: "f2<&constMsg[0]>", linkageName: "_Z2f2IXadsoKcL_ZL8constMsgEEEEvv",
// CHECK: ![[LOC36]] = !DILocation(line: 0, scope: ![[SUBPROG26]], inlinedAt: ![[LOC37:.*]])
// CHECK: ![[LOC37]] = !DILocation(line: 44, column: 3, scope: ![[SUBPROG32]])

char const constMsg[] = "hello";

void f0() {
  __builtin_verbose_trap("Argument_must_not_be_null");
}

void f1() {
  __builtin_verbose_trap("Argument_must_not_be_null");
  __builtin_verbose_trap("hello");
}

template <const char * const str>
void f2() {
  __builtin_verbose_trap(str);
}

void f3() {
  f2<constMsg>();
}
