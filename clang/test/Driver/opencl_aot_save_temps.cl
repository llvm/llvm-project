// RUN: %clang -x cl --save-temps -c -### %s 2>&1 | FileCheck %s
// RUN: %clang -x cl -ccc-print-phases -c %s 2>&1 | FileCheck %s -check-prefix=CHECK-PHASES

// CHECK: "-o" "[[CLI_NAME:.+]].cli" "-x" "cl"
// CHECK-NEXT:  "-o" "[[CLI_NAME]].bc" "-x" "cl-cpp-output"{{.*}}"[[CLI_NAME:.+]].cli"

// CHECK-PHASES: 0: input, {{.*}}, cl
// CHECK-PHASES: 1: preprocessor, {0}, cl-cpp-output
// CHECK-PHASES: 2: compiler, {1}, ir

uint3 add(uint3 a, uint3 b) {
  ulong x = a.x + (ulong)b.x;
  ulong y = a.y + (ulong)b.y + (x >> 32);
  uint z = a.z + b.z + (y >> 32);
  return (uint3)(x, y, z);
}
