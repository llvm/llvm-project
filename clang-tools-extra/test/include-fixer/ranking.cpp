// RUN: clang-include-fixer -db=yaml -input=%S/Inputs/fake_yaml_db.yaml -output-headers %s -- | FileCheck %s

// CHECK:     "HeaderInfos": [
// CHECK-NEXT:  {"Header": "\"../include/bar.h\"",
// CHECK-NEXT:   "QualifiedName": "b::a::bar"},
// CHECK-NEXT:  {"Header": "\"../include/zbar.h\"",
// CHECK-NEXT:   "QualifiedName": "b::a::bar"}
// CHECK-NEXT:]

bar b;
