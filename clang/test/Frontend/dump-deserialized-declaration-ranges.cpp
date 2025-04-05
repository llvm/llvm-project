// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -xc++ -fmodules -fmodule-name=foo -fmodule-map-file=%t/foo.cppmap -emit-module %t/foo.cppmap -o %t/foo.pcm
// RUN: %clang_cc1 -xc++ -fmodules -dump-deserialized-declaration-ranges=%t/decls -fmodule-file=%t/foo.pcm %t/foo.cpp -o %t/foo.o
// RUN: cat %t/decls | FileCheck -check-prefix=RANGE %s
// RANGE:{
// RANGE-NEXT:  "required_ranges": [
// RANGE-NEXT:    {
// RANGE-NEXT:      "file": "{{.+}}/foo.h",
// RANGE-NEXT:      "range": [
// RANGE-NEXT:        {
// RANGE-NEXT:          "from": {
// RANGE-NEXT:            "line": 1,
// RANGE-NEXT:            "column": 1
// RANGE-NEXT:          },
// RANGE-NEXT:          "to": {
// RANGE-NEXT:            "line": 9,
// RANGE-NEXT:            "column": 1
// RANGE-NEXT:          }
// RANGE-NEXT:        },
// RANGE-NEXT:        {
// RANGE-NEXT:          "from": {
// RANGE-NEXT:            "line": 11,
// RANGE-NEXT:            "column": 1
// RANGE-NEXT:          },
// RANGE-NEXT:          "to": {
// RANGE-NEXT:            "line": 11,
// RANGE-NEXT:            "column": 12
// RANGE-NEXT:          }
// RANGE-NEXT:        },
// RANGE-NEXT:        {
// RANGE-NEXT:          "from": {
// RANGE-NEXT:            "line": 13,
// RANGE-NEXT:            "column": 1
// RANGE-NEXT:          },
// RANGE-NEXT:          "to": {
// RANGE-NEXT:            "line": 15,
// RANGE-NEXT:            "column": 1
// RANGE-NEXT:          }
// RANGE-NEXT:        }
// RANGE-NEXT:      ]
// RANGE-NEXT:    }
// RANGE-NEXT:  ]
// RANGE-NEXT:}
// RUN: echo -e '{\n\
// RUN:  "required_ranges": [\n\
// RUN:    {\n\
// RUN:      "file": "%t/foo.h",\n\
// RUN:      "range": [\n\
// RUN:        {\n\
// RUN:          "from": {\n\
// RUN:            "line": 1,\n\
// RUN:            "column": 1\n\
// RUN:          },\n\
// RUN:          "to": {\n\
// RUN:            "line": 9,\n\
// RUN:            "column": 1\n\
// RUN:          }\n\
// RUN:        },\n\
// RUN:        {\n\
// RUN:          "from": {\n\
// RUN:            "line": 11,\n\
// RUN:            "column": 1\n\
// RUN:          },\n\
// RUN:          "to": {\n\
// RUN:            "line": 11,\n\
// RUN:            "column": 12\n\
// RUN:          }\n\
// RUN:        },\n\
// RUN:        {\n\
// RUN:          "from": {\n\
// RUN:            "line": 13,\n\
// RUN:            "column": 1\n\
// RUN:          },\n\
// RUN:          "to": {\n\
// RUN:            "line": 15,\n\
// RUN:            "column": 1\n\
// RUN:          }\n\
// RUN:        }\n\
// RUN:      ]\n\
// RUN:    }\n\
// RUN:  ]\n\
// RUN:}' > %t/expected_decls
// RUN: diff %t/decls %t/expected_decls

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
