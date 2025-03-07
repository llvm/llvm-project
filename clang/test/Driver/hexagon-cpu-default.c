// CHECK: "-target-cpu" "hexagonv68"

// RUN: %clang -c %s -### --target=hexagon-unknown-elf \
// RUN:  2>&1 | FileCheck  %s
