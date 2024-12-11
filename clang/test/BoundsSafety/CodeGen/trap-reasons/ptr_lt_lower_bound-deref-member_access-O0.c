
// RUN: %clang_cc1 -O0 -debug-info-kind=standalone -dwarf-version=5 -fbounds-safety -emit-llvm %s -o %t_bidi.ll
// RUN: echo "; __SEPERATOR__" > %t.sep
// RUN: cat %t_bidi.ll %t.sep %t_bidi.ll > %t_bidi.repeated.ll
// RUN: FileCheck %s --input-file=%t_bidi.repeated.ll
#include <ptrcheck.h>

typedef struct Data {
    int a;
    int b;
} Data_t;

// Don't need to test `__indexable` because there is no lower bound check

int operation(Data_t* __bidi_indexable d) {
    return d->a;
}

// In first copy of the file
// CHECK-DAG: [[OPT_REMARK:![0-9]+]] = !{!"bounds-safety-check-ptr-ge-lower-bound"}
// CHECK-DAG: [[TRAP_SCOPE:![0-9]+]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$Dereferencing below bounds", scope: [[FILE_SCOPE:![0-9]+]], file: [[FILE_SCOPE:![0-9]+]], type: {{![0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: {{![0-9]+}}
// CHECK-DAG: [[TRAP_LOC:![0-9]+]] = !DILocation(line: 0, scope: [[TRAP_SCOPE]], inlinedAt: [[SRC_LOC:![0-9]+]])
// CHECK-DAG: [[SRC_LOC]] = !DILocation(line: 16, column: 15, scope: {{![0-9]+}}) 
// CHECK-LABEL: ; __SEPERATOR__

// In second copy of the file
// CHECK-LABEL: @operation
// Note: We use the `OPT_REMARK` to make sure we match against the correct branch.
// Unfortunately `SRC_LOC` is not enough because it isn't unique.
// CHECK: br i1 %{{.+}}, label %{{.+}}, label %[[TRAP_LABEL:[a-z0-9]+]], !dbg [[SRC_LOC]], !annotation [[OPT_REMARK]]
// CHECK: [[TRAP_LABEL]]:
// CHECK-NEXT: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[TRAP_LOC]]
