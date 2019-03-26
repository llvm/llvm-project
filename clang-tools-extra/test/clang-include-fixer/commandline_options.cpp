// RUN: echo "foo f;" > %t.cpp
// RUN: clang-include-fixer -db=fixed -input='foo= "foo.h","bar.h"' -output-headers %t.cpp -- | FileCheck %s
// RUN: cat %t.cpp | clang-include-fixer -stdin -insert-header='{FilePath: "%/t.cpp", QuerySymbolInfos: [{RawIdentifier: foo, Range: {Offset: 0, Length: 3}}], HeaderInfos: [{Header: "\"foo.h\"", QualifiedName: "foo"}]}' %t.cpp | FileCheck %s -check-prefix=CHECK-CODE
// RUN: cat %t.cpp | not clang-include-fixer -stdin -insert-header='{FilePath: "%/t.cpp", QuerySymbolInfos: [{RawIdentifier: foo, Range: {Offset: 0, Length: 3}}], HeaderInfos: [{Header: "\"foo.h\"", QualifiedName: "foo"},{Header: "\"foo2.h\"", QualifiedName: "foo"}]}' %t.cpp
// RUN: cat %t.cpp | clang-include-fixer -stdin -insert-header='{FilePath: "%/t.cpp", QuerySymbolInfos: [{RawIdentifier: foo, Range: {Offset: 0, Length: 3}}], HeaderInfos: [{Header: "\"foo.h\"", QualifiedName: "a:foo"},{Header: "\"foo.h\"", QualifiedName: "b:foo"}]}' %t.cpp
//
// CHECK:     "HeaderInfos": [
// CHECK-NEXT:  {"Header": "\"foo.h\"",
// CHECK-NEXT:   "QualifiedName": "foo"},
// CHECK-NEXT:  {"Header": "\"bar.h\"",
// CHECK-NEXT:   "QualifiedName": "foo"}
// CHECK-NEXT:]
//
// CHECK-CODE: #include "foo.h"
// CHECK-CODE: foo f;
