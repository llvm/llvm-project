// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -xc++ -fmodules -fmodule-name=foo -fmodule-map-file=%t/foo.cppmap -emit-module %t/foo.cppmap -o %t/foo.pcm
// RUN: %clang_cc1 -xc++ -fmodules -dump-minimization-hints=%t/decls -fmodule-file=%t/foo.pcm %t/foo.cpp -o %t/foo.o
// RUN: cat %t/decls
// RUN: cat %t/decls | FileCheck -check-prefix=RANGE %s
// RANGE:{
// RANGE-NEXT:  "required_ranges": [
// RANGE-NEXT:    {
// RANGE-NEXT:      "file": "{{.+}}foo.h",
// RANGE-NEXT:      "range": [
// RANGE-NEXT:        {
// RANGE-NEXT:          "from": {
// RANGE-NEXT:            "line": 1,
// RANGE-NEXT:            "column": 1
// RANGE-NEXT:          },
// RANGE-NEXT:          "to": {
// RANGE-NEXT:            "line": 9,
// RANGE-NEXT:            "column": 3
// RANGE-NEXT:          }
// RANGE-NEXT:        },
// RANGE-NEXT:        {
// RANGE-NEXT:          "from": {
// RANGE-NEXT:            "line": 11,
// RANGE-NEXT:            "column": 1
// RANGE-NEXT:          },
// RANGE-NEXT:          "to": {
// RANGE-NEXT:            "line": 11,
// RANGE-NEXT:            "column": 25
// RANGE-NEXT:          }
// RANGE-NEXT:        },
// RANGE-NEXT:        {
// RANGE-NEXT:          "from": {
// RANGE-NEXT:            "line": 13,
// RANGE-NEXT:            "column": 1
// RANGE-NEXT:          },
// RANGE-NEXT:          "to": {
// RANGE-NEXT:            "line": 15,
// RANGE-NEXT:            "column": 2
// RANGE-NEXT:          }
// RANGE-NEXT:        }
// RANGE-NEXT:      ]
// RANGE-NEXT:    }
// RANGE-NEXT:  ]
// RANGE-NEXT:}

//--- foo.cppmap
module foo {
  header "foo.h"
  export *
}

//--- foo.h
class MyData {
public:
    MyData(int val): value_(val) {}
    int getValue() const {
        return 5;
    }
private:
    int value_;
};

extern int global_value;

int multiply(int a, int b) {
    return a * b;
}

//--- foo.cpp
#include "foo.h"
int global_value = 5;
int main() {
  MyData data(5);
  int current_value = data.getValue();
  int doubled_value = multiply(current_value, 2);
  int final_result = doubled_value + global_value;
}
