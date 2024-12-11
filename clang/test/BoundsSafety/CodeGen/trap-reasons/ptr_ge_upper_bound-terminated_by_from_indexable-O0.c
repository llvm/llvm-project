
// RUN: %clang_cc1 -O0 -debug-info-kind=standalone -dwarf-version=5 -fbounds-safety -emit-llvm %s -o %t.ll
// RUN: echo "; __SEPERATOR__" > %t.sep
// RUN: cat %t.ll %t.sep %t.ll > %t.repeated.ll
// RUN: FileCheck %s --input-file=%t.repeated.ll
#include <ptrcheck.h>

void convert(const char* __indexable str) {
    const char* __null_terminated convert = __unsafe_null_terminated_from_indexable(str);
}

// We effectively need to walk the IR backwards starting from the debug info
// and then locating the expected IR instructions. We don't have a good way
// of walking the file backwards other than make everything a CHECK-DAG.
// That would allow many different orderings rather than the one we want, which
// isn't ideal. Instead make the input to FileCheck be two copies of the IR so
// that we can first match the debug info and then the IR instructions.

// In first copy of the file
// CHECK-DAG: [[OPT_REMARK:![0-9]+]] = !{!"bounds-safety-check-ptr-lt-upper-bound"}
// CHECK-DAG: [[TRAP_SCOPE:![0-9]+]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$Pointer above bounds while converting __indexable to __terminated_by", scope: [[FILE_SCOPE:![0-9]+]], file: [[FILE_SCOPE:![0-9]+]], type: {{![0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: {{![0-9]+}}
// CHECK-DAG: [[TRAP_LOC:![0-9]+]] = !DILocation(line: 0, scope: [[TRAP_SCOPE]], inlinedAt: [[SRC_LOC:![0-9]+]])
// CHECK-LABEL: ; __SEPERATOR__

// In second copy of the file
// CHECK-LABEL: @convert
// Note: We use the `OPT_REMARK` to make sure we match against the correct branch.
// Unfortunately `SRC_LOC` is not enough because it isn't unique.
// CHECK: br i1 %{{.+}}, label %{{.+}}, label %[[TRAP_LABEL:[a-z0-9]+]], !dbg [[SRC_LOC]], !annotation [[OPT_REMARK]]
// CHECK: [[TRAP_LABEL]]:
// CHECK-NEXT: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[TRAP_LOC]]
// CHECK: [[SRC_LOC]] = !DILocation(line: 9, column: 45, scope: {{![0-9]+}})
