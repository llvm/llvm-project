// RUN: clang-include-fixer -db=yaml -input=%S/Inputs/fake_yaml_db.yaml -output-headers %s -- | FileCheck %s -implicit-check-not=.h

// CHECK: "../include/bar.h"
// CHECK-NEXT: "../include/zbar.h"

bar b;
