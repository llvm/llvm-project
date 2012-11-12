struct Point {
  float x;
  float y;
  float z;
};

#define MACRO2(x) x
#define MACRO(x) MACRO2(x)

void test(struct Point *p) {
        p->x;
  MACRO(p->x);
}

// RUN: c-index-test -code-completion-at=%s:11:12 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:12:12 %s | FileCheck %s
// CHECK:      FieldDecl:{ResultType float}{TypedText x} (35)
// CHECK-NEXT: FieldDecl:{ResultType float}{TypedText y} (35)
// CHECK-NEXT: FieldDecl:{ResultType float}{TypedText z} (35)
// CHECK-NEXT: Completion contexts:
// CHECK-NEXT: Arrow member access
// CHECK-NEXT: Container Kind: StructDecl
