// RUN: clang-include-fixer -db=yaml -input=%S/Inputs/fake_yaml_db.yaml -output-headers %s -- | FileCheck %s

// CHECK: "Headers": [ "\"../include/bar.h\"", "\"../include/zbar.h\"" ]

bar b;
