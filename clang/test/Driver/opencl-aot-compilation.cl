// RUN: %clang -x cl %s --save-temps -### 2>&1 | FileCheck %s

// CHECK: "-fdeclare-opencl-builtins" {{.*}} "[[SRC:.+]].cli" "-x" "cl" "{{.*}}[[SRC]].cl"
// CHECK-NEXT: "-fdeclare-opencl-builtins" {{.*}} "-x" "cl-cpp-output" "[[SRC]].cli"
