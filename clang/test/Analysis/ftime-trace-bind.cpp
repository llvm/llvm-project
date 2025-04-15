// RUN: %clang_analyze_cc1 -analyzer-checker=core %s -ftime-trace=%t.raw.json -ftime-trace-granularity=0 -verify
// RUN: %python -c 'import json, sys; print(json.dumps(json.load(sys.stdin), indent=4))' < %t.raw.json > %t.formatted.json
// RUN: FileCheck --input-file=%t.formatted.json --check-prefix=CHECK %s

// CHECK:          "name": "RegionStoreManager::bindArray",
// CHECK-NEXT:     "args": {
//
// The below does not necessarily follow immediately,
// depending on what parts of the array are initialized first.
//
// CHECK:              "detail": "'arr[0][1]'"
// CHECK-NEXT:     }
//
// CHECK:              "detail": "'arr[0]'"
// CHECK-NEXT:     }
//
// CHECK:              "detail": "'arr'"
// CHECK-NEXT:     }

int f() {
    int arr[2][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    return arr[1][0][1];
}
// expected-no-diagnostics
