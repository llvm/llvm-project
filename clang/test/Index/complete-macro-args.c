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

#define MACRO3(x,y,z) x;y;z

void test2(struct Point *p) {
  MACRO3(p->x);
  MACRO3(p->x
}

#define FM(x) x
void test3(struct Point *p) {
  FM(p->x, a);
}

#define VGM(...) 0
#define VGM2(...) __VA_ARGS__

// These need to be last, to test proper handling of EOF.
#ifdef EOF_TEST1
void test3(struct Point *p) {
  VGM(1,2, p->x

#elif EOF_TEST2
void test3(struct Point *p) {
  VGM2(VGM(1,2, p->x

#endif

// RUN: c-index-test -code-completion-at=%s:11:12 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:12:12 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:18:13 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:19:13 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:24:9 %s | FileCheck %s
// CHECK:      FieldDecl:{ResultType float}{TypedText x} (35)
// CHECK-NEXT: FieldDecl:{ResultType float}{TypedText y} (35)
// CHECK-NEXT: FieldDecl:{ResultType float}{TypedText z} (35)
// CHECK-NEXT: Completion contexts:
// CHECK-NEXT: Arrow member access
// CHECK-NEXT: Container Kind: StructDecl

// With these, code-completion is unknown because the macro argument (and the
// completion point) is not expanded by the macro definition.
// RUN: c-index-test -code-completion-at=%s:33:15 %s -DEOF_TEST1 | FileCheck %s -check-prefix=CHECK-EOF
// RUN: c-index-test -code-completion-at=%s:37:20 %s -DEOF_TEST2 | FileCheck %s -check-prefix=CHECK-EOF
// CHECK-EOF: Completion contexts:
// CHECK-EOF: Unknown
