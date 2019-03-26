// RUN: clang-include-fixer -db=fixed -input='foo= "foo.h","bar.h"' -query-symbol="foo" test.cpp -- | FileCheck %s

// CHECK:     "FilePath": "test.cpp",
// CHECK-NEXT:"QuerySymbolInfos": [
// CHECK-NEXT:   {"RawIdentifier": "foo",
// CHECK-NEXT:    "Range":{"Offset":0,"Length":0}}
// CHECK-NEXT:],
// CHECK-NEXT:"HeaderInfos": [
// CHECK-NEXT:  {"Header": "\"foo.h\"",
// CHECK-NEXT:   "QualifiedName": "foo"},
// CHECK-NEXT:  {"Header": "\"bar.h\"",
// CHECK-NEXT:   "QualifiedName": "foo"}
// CHECK-NEXT:]
