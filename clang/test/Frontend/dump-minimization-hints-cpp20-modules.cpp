// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++20 %t/foo.cppm -emit-module-interface -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -dump-minimization-hints=%t/decls
// RUN: cat %t/decls
// RUN: cat %t/decls | FileCheck -check-prefix=RANGE %s
// RANGE:{
// RANGE-NEXT:"required_ranges": [
// RANGE-NEXT:  {
// RANGE-NEXT:    "file": "{{.+}}foo.cppm",
// RANGE-NEXT:    "range": [
// RANGE-NEXT:      {
// RANGE-NEXT:        "from": {
// RANGE-NEXT:          "line": 3,
// RANGE-NEXT:          "column": 1
// RANGE-NEXT:        },
// RANGE-NEXT:        "to": {
// RANGE-NEXT:          "line": 3,
// RANGE-NEXT:          "column": 22
// RANGE-NEXT:        }
// RANGE-NEXT:      },
// RANGE-NEXT:      {
// RANGE-NEXT:        "from": {
// RANGE-NEXT:          "line": 4,
// RANGE-NEXT:          "column": 3
// RANGE-NEXT:        },
// RANGE-NEXT:        "to": {
// RANGE-NEXT:          "line": 4,
// RANGE-NEXT:          "column": 9
// RANGE-NEXT:        }
// RANGE-NEXT:      },
// RANGE-NEXT:      {
// RANGE-NEXT:        "from": {
// RANGE-NEXT:          "line": 4,
// RANGE-NEXT:          "column": 10
// RANGE-NEXT:        },
// RANGE-NEXT:        "to": {
// RANGE-NEXT:          "line": 4,
// RANGE-NEXT:          "column": 43
// RANGE-NEXT:        }
// RANGE-NEXT:      },
// RANGE-NEXT:      {
// RANGE-NEXT:        "from": {
// RANGE-NEXT:          "line": 6,
// RANGE-NEXT:          "column": 1
// RANGE-NEXT:        },
// RANGE-NEXT:        "to": {
// RANGE-NEXT:          "line": 6,
// RANGE-NEXT:          "column": 2
// RANGE-NEXT:        }
// RANGE-NEXT:      },
// RANGE-NEXT:      {
// RANGE-NEXT:        "from": {
// RANGE-NEXT:          "line": 8,
// RANGE-NEXT:          "column": 1
// RANGE-NEXT:        },
// RANGE-NEXT:        "to": {
// RANGE-NEXT:          "line": 8,
// RANGE-NEXT:          "column": 7
// RANGE-NEXT:        }
// RANGE-NEXT:      },
// RANGE-NEXT:      {
// RANGE-NEXT:        "from": {
// RANGE-NEXT:          "line": 8,
// RANGE-NEXT:          "column": 8
// RANGE-NEXT:        },
// RANGE-NEXT:        "to": {
// RANGE-NEXT:          "line": 8,
// RANGE-NEXT:          "column": 25
// RANGE-NEXT:        }
// RANGE-NEXT:      },
// RANGE-NEXT:      {
// RANGE-NEXT:        "from": {
// RANGE-NEXT:          "line": 9,
// RANGE-NEXT:          "column": 3
// RANGE-NEXT:        },
// RANGE-NEXT:        "to": {
// RANGE-NEXT:          "line": 9,
// RANGE-NEXT:          "column": 36
// RANGE-NEXT:        }
// RANGE-NEXT:      },
// RANGE-NEXT:      {
// RANGE-NEXT:        "from": {
// RANGE-NEXT:          "line": 11,
// RANGE-NEXT:          "column": 1
// RANGE-NEXT:        },
// RANGE-NEXT:        "to": {
// RANGE-NEXT:          "line": 11,
// RANGE-NEXT:          "column": 2
// RANGE-NEXT:        }
// RANGE-NEXT:      }
// RANGE-NEXT:    ]
// RANGE-NEXT:  }
// RANGE-NEXT:]
// RANGE-NEXT:}

//--- foo.cppm
export module foo;

namespace piecemeal { // line 3
  export int used(int n) { return n + 1; }
  export int unused(int n) { return n + 2; }
}

export namespace whole { // line 8
  int used(int n) { return n + 1; }
  int unused(int n) { return n + 3; }
} // line 11

//--- use.cpp
import foo;

int main() {
  piecemeal::used(4);  // only one of the functions used from each namespace.
  whole::used(4);
}
