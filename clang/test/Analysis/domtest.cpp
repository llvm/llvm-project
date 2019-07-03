// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=debug.DumpDominators \
// RUN:   2>&1 | FileCheck %s

bool coin();

namespace pr42041_unreachable_cfg_successor {
enum Kind {
  A
};

void f() {
  switch(Kind{}) {
  case A:
    break;
  }
}
} // end of namespace pr42041_unreachable_cfg_successor

//  [B3 (ENTRY)]  -> [B1] -> [B2] -> [B0 (EXIT)]

// CHECK:      Immediate dominance tree (Node#,IDom#):
// CHECK-NEXT: (0,2)
// CHECK-NEXT: (1,3)
// CHECK-NEXT: (2,1)
// CHECK-NEXT: (3,3)

void funcWithBranch() {
  int x = 0;
  if (coin()) {
    if (coin()) {
      x = 5;
    }
    int j = 10 / x;
    (void)j;
  }
}

//                            ----> [B2] ---->
//                           /                \
// [B5 (ENTRY)] -> [B4] -> [B3] -----------> [B1]
//                   \                       /
//                    ----> [B0 (EXIT)] <----

// CHECK:      Immediate dominance tree (Node#,IDom#):
// CHECK-NEXT: (0,4)
// CHECK-NEXT: (1,3)
// CHECK-NEXT: (2,3)
// CHECK-NEXT: (3,4)
// CHECK-NEXT: (4,5)
// CHECK-NEXT: (5,5)
