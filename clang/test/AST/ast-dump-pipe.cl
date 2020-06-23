// Test without serialization:
// RUN: %clang_cc1 -triple spir64 -cl-std=CL2.0 -ast-dump -ast-dump-filter pipetype %s \
// RUN: | FileCheck -strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple spir64 -cl-std=CL2.0 -emit-pch -o %t %s
// RUN: %clang_cc1 -x cl -triple spir64 -cl-std=CL2.0 -include-pch %t -ast-dump-all -ast-dump-filter pipetype /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck -strict-whitespace %s

typedef pipe int pipetype;
// CHECK:      PipeType {{.*}} 'read_only pipe int'
// CHECK-NEXT:   BuiltinType {{.*}} 'int'

typedef read_only pipe int pipetype2;
// CHECK:      PipeType {{.*}} 'read_only pipe int'
// CHECK-NEXT:   BuiltinType {{.*}} 'int'

typedef write_only pipe int pipetype3;
// CHECK:      PipeType {{.*}} 'write_only pipe int'
// CHECK-NEXT:   BuiltinType {{.*}} 'int'
