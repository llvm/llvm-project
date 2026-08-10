// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s

struct MyStruct {
  int myfield;
};

enum {
  MyValToSet
};

// CHECK: [[@LINE+1]]:14 | struct/C | MyStruct |
const struct MyStruct _MyStruct[]
  [16]
  [3]
  [3]
  [2]
  [2] = {
 [0] = {
    [0] = {
      [0] = {
        [0][0] = {
          [0] = {
            // CHECK: [[@LINE+2]]:14 | field/C | myfield | {{.*}} | Ref,RelCont |
            // CHECK: [[@LINE+1]]:24 | enumerator/C | MyValToSet |
            .myfield = MyValToSet,
            // CHECK-NOT: | field/C | myfield |
            // CHECK-NOT: | enumerator/C | MyValToSet |
          },
        },
      },
    },
  },
};
