// RUN: %clang_cc1 -triple spir64 -cl-std=CL2.0 -ast-dump -ast-dump-filter pipetype %s | FileCheck -strict-whitespace %s
typedef pipe int pipetype;
// CHECK:      PipeType {{.*}} 'pipe int'
// CHECK-NEXT:   BuiltinType {{.*}} 'int'
