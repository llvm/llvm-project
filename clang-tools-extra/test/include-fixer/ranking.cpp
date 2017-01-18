// RUN: clang-include-fixer -db=yaml -input=%S/Inputs/fake_yaml_db.yaml -output-headers %s -- | FileCheck %s
// RUN: clang-include-fixer -query-symbol bar -db=yaml -input=%S/Inputs/fake_yaml_db.yaml -output-headers %s -- | FileCheck %s

// CHECK:     "HeaderInfos": [
// CHECK-NEXT:  {"Header": "\"test/include-fixer/baz.h\"",
// CHECK-NEXT:   "QualifiedName": "c::bar"},
// CHECK-NEXT:  {"Header": "\"../include/bar.h\"",
// CHECK-NEXT:   "QualifiedName": "b::a::bar"},
// CHECK-NEXT:  {"Header": "\"../include/zbar.h\"",
// CHECK-NEXT:   "QualifiedName": "b::a::bar"}
// CHECK-NEXT:]

bar b;
