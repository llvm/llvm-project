// RUN: %clang_cc1 -triple x86_64-pc-linux -std=c++2a -ast-dump=json %s | FileCheck %s

namespace foo {
}
// CHECK: "kind": "NamespaceDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 3
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 3
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 4
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "foo"
// CHECK-NEXT: },


namespace {
}
// CHECK: "kind": "NamespaceDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 27
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 27
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 28
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "UsingDirectiveDecl",
// CHECK-NEXT: "loc": {},
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 27
// CHECK-NEXT: },
// CHECK-NEXT: "end": {}
// CHECK-NEXT: },
// CHECK-NEXT: "isImplicit": true,
// CHECK-NEXT: "nominatedNamespace": {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "NamespaceDecl",
// CHECK-NEXT: "name": ""
// CHECK-NEXT: }
// CHECK-NEXT: },

namespace bar {
inline namespace __1 {
}
}
// CHECK: "kind": "NamespaceDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 68
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 68
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 71
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "bar",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT:   "id": "0x{{.*}}",
// CHECK-NEXT:   "kind": "NamespaceDecl",
// CHECK-NEXT:   "loc": {
// CHECK-NEXT:     "col": 18,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 69
// CHECK-NEXT:   },
// CHECK-NEXT:   "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:       "col": 1,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 69
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:       "col": 1,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 70
// CHECK-NEXT:     }
// CHECK-NEXT:   },
// CHECK-NEXT:   "name": "__1",
// CHECK-NEXT:   "isInline": true
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },

namespace baz::quux {
}
// CHECK: "kind": "NamespaceDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 118
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 118
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 119
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "baz",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT:   "id": "0x{{.*}}",
// CHECK-NEXT:   "kind": "NamespaceDecl",
// CHECK-NEXT:   "loc": {
// CHECK-NEXT:     "col": 16,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 118
// CHECK-NEXT:   },
// CHECK-NEXT:   "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:       "col": 14,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 118
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:       "col": 1,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 119
// CHECK-NEXT:     }
// CHECK-NEXT:   },
// CHECK-NEXT:   "name": "quux"
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },

namespace quux::inline frobble {
}
// CHECK: "kind": "NamespaceDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 165
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 165
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 166
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "quux",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT:   "id": "0x{{.*}}",
// CHECK-NEXT:   "kind": "NamespaceDecl",
// CHECK-NEXT:   "loc": {
// CHECK-NEXT:     "col": 24,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 165
// CHECK-NEXT:   },
// CHECK-NEXT:   "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:       "col": 17,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 165
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:       "col": 1,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 166
// CHECK-NEXT:     }
// CHECK-NEXT:   },
// CHECK-NEXT:   "name": "frobble",
// CHECK-NEXT:   "isInline": true
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
