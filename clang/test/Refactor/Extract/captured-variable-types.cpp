
typedef struct {
  int width, height;
} Rectangle;

int basicTypes(int i, float f, char c, const int *ip, float *fp,
               const Rectangle *structPointer) {
  return basicTypes(i, f, c, ip, fp, structPointer);
}
// CHECK1: "static int extracted(char c, float f, float *fp, int i, const int *ip, const Rectangle *structPointer) {\nreturn basicTypes(i, f, c, ip, fp, structPointer);\n}\n\n" [[@LINE-4]]:1 -> [[@LINE-4]]:1
// CHECK1-NEXT: "extracted(c, f, fp, i, ip, structPointer)" [[@LINE-3]]:10 -> [[@LINE-3]]:52

// RUN: clang-refactor-test perform -action extract -selected=%s:8:10-8:52 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:8:10-8:52 %s -x c | FileCheck --check-prefix=CHECK1 %s

#ifndef __cplusplus
#define bool _Bool
#define true 1
#define false 0
#endif

int boolType(bool b) {
  bool b2 = true;
  return boolType(b && b2 && true && false);
}
// CHECK2: "static int extracted(bool b, bool b2) {\nreturn boolType(b && b2 && true && false);\n}\n\n" [[@LINE-4]]:1 -> [[@LINE-4]]:1
// CHECK2-NEXT: "extracted(b, b2)" [[@LINE-3]]:10 -> [[@LINE-3]]:44

// RUN: clang-refactor-test perform -action extract -selected=%s:24:10-24:44 %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:24:10-24:44 %s -x c | FileCheck --check-prefix=CHECK2 %s

int global = 0;

void dontCaptureGlobalVariable() {
  static int staticInFunction = 0;
  int i = global + staticInFunction;
}
// CHECK3: "static int extracted(int staticInFunction) {\nreturn global + staticInFunction;\n}\n\n" [[@LINE-4]]:1 -> [[@LINE-4]]:1
// CHECK3-NEXT: "extracted(staticInFunction)" [[@LINE-3]]:11 -> [[@LINE-3]]:36

// RUN: clang-refactor-test perform -action extract -selected=%s:36:11-36:36 %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:36:11-36:36 %s -x c | FileCheck --check-prefix=CHECK3 %s

#ifdef __cplusplus

int referenceType(const Rectangle &r, int &i) {
  return referenceType(r, i);
}
// CHECK4: "static int extracted(int &i, const Rectangle &r) {\nreturn referenceType(r, i);\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK4-NEXT: "extracted(i, r)" [[@LINE-3]]:10 -> [[@LINE-3]]:29

#endif

// RUN: clang-refactor-test perform -action extract -selected=%s:47:10-47:29 %s | FileCheck --check-prefix=CHECK4 %s

typedef union {
  int x;
  Rectangle r;
} Union;

int aggregateTypeNotMutated(Rectangle r, Union u) {
  Rectangle r2 = r;
  Union u2 = u;
  aggregateTypeNotMutated(r, u);
  return aggregateTypeNotMutated(r2, u2);
}
// CHECK5-CPP: "static int extracted(const Rectangle &r, const Union &u) {\nreturn aggregateTypeNotMutated(r, u);\n}\n\n" [[@LINE-6]]:1 -> [[@LINE-6]]:1
// CHECK5-CPP-NEXT: "extracted(r, u)" [[@LINE-4]]:3 -> [[@LINE-4]]:32
// CHECK5-C: "static int extracted(const Rectangle *r, const Union *u) {\nreturn aggregateTypeNotMutated(*r, *u);\n}\n\n" [[@LINE-8]]:1 -> [[@LINE-8]]:1
// CHECK5-C-NEXT: "extracted(&r, &u)" [[@LINE-6]]:3 -> [[@LINE-6]]:32
// CHECK5-CPP: "static int extracted(const Rectangle &r2, const Union &u2) {\nreturn aggregateTypeNotMutated(r2, u2);\n}\n\n" [[@LINE-10]]:1 -> [[@LINE-10]]:1
// CHECK5-CPP-NEXT: "extracted(r2, u2)" [[@LINE-7]]:10 -> [[@LINE-7]]:41
// CHECK5-C: "static int extracted(const Rectangle *r2, const Union *u2) {\nreturn aggregateTypeNotMutated(*r2, *u2);\n}\n\n" [[@LINE-12]]:1 -> [[@LINE-12]]:1
// CHECK5-C-NEXT: "extracted(&r2, &u2)" [[@LINE-9]]:10 -> [[@LINE-9]]:41

// RUN: clang-refactor-test perform -action extract -selected=%s:64:3-64:32 -selected=%s:65:10-65:41 %s | FileCheck --check-prefix=CHECK5-CPP %s
// RUN: clang-refactor-test perform -action extract -selected=%s:64:3-64:32 -selected=%s:65:10-65:41 %s -x c | FileCheck --check-prefix=CHECK5-C %s
;
int arrayType(const int x[]) {
  int v[2] = {1, 2};
  arrayType(x);
  return arrayType(v);
}
// CHECK6: "static int extracted(const int *x) {\nreturn arrayType(x);\n}\n\n" [[@LINE-5]]:1 -> [[@LINE-5]]:1
// CHECK6-NEXT: "extracted(x)" [[@LINE-4]]:3 -> [[@LINE-4]]:15
// CHECK6: "static int extracted(int *v) {\nreturn arrayType(v);\n}\n\n" [[@LINE-7]]:1 -> [[@LINE-7]]:1
// CHECK6-NEXT: "extracted(v)" [[@LINE-5]]:10 -> [[@LINE-5]]:22

// RUN: clang-refactor-test perform -action extract -selected=%s:81:3-81:15 -selected=%s:82:10-82:22 %s | FileCheck --check-prefix=CHECK6 %s

typedef enum {
  EnumA, EnumB
} Enum;

int enumType(Enum e) {
  return enumType(e);
}
// CHECK7: "static int extracted(Enum e) {\nreturn enumType(e);\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK7-NEXT: "extracted(e)" [[@LINE-3]]:10 -> [[@LINE-3]]:21

// RUN: clang-refactor-test perform -action extract -selected=%s:96:10-96:21 %s | FileCheck --check-prefix=CHECK7 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:96:10-96:21 %s -x c | FileCheck --check-prefix=CHECK7 %s
;
int qualifierReduction(const int i, const int * const p, int *volatile pv, int *__restrict__ rv,
                       const Enum e) {
  return qualifierReduction(i, p, pv, rv, e);
}
// CHECK8: "static int extracted(Enum e, int i, const int *p, int *volatile pv, int *__restrict rv) {\nreturn qualifierReduction(i, p, pv, rv, e);\n}\n\n" [[@LINE-4]]:1 -> [[@LINE-4]]:1
// CHECK8-NEXT: "extracted(e, i, p, pv, rv)" [[@LINE-3]]:10 -> [[@LINE-3]]:45

// RUN: clang-refactor-test perform -action extract -selected=%s:106:10-106:45 %s | FileCheck --check-prefix=CHECK8 %s

#ifdef __cplusplus

int autoTypeHandling(int x, Rectangle &ref) {
  auto i = x;
  auto &r = ref;
  return autoTypeHandling(i, r);
}
// CHECK9: "static int extracted(int i, Rectangle &r) {\nreturn autoTypeHandling(i, r);\n}\n\n" [[@LINE-5]]:1 -> [[@LINE-5]]:1
// CHECK9-NEXT: "extracted(i, r)" [[@LINE-3]]:10 -> [[@LINE-3]]:32

#endif

// RUN: clang-refactor-test perform -action extract -selected=%s:118:10-118:32 %s | FileCheck --check-prefix=CHECK9 %s

Rectangle globalRect;
int capturedVariableMutation(int x, int y, int z, const int *ip, Rectangle r) {
  return x = 0, y += 2, z &= 3, ip = 0, r = globalRect, capturedVariableMutation(x, 1, 2, ip, r);
}
// CHECK10-CPP: "static int extracted(const int *&ip, Rectangle &r, int &x, int &y, int &z) {\nreturn x = 0, y += 2, z &= 3, ip = 0, r = globalRect, capturedVariableMutation(x, 1, 2, ip, r);\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK10-CPP-NEXT: "extracted(ip, r, x, y, z)" [[@LINE-3]]:10 -> [[@LINE-3]]:97

// CHECK10-C: "static int extracted(const int **ip, Rectangle *r, int *x, int *y, int *z) {\nreturn *x = 0, *y += 2, *z &= 3, *ip = 0, *r = globalRect, capturedVariableMutation(*x, 1, 2, *ip, *r);\n}\n\n" [[@LINE-6]]:1 -> [[@LINE-6]]:1
// CHECK10-C-NEXT: "extracted(&ip, &r, &x, &y, &z)" [[@LINE-6]]:10 -> [[@LINE-6]]:97

#ifdef __cplusplus

int capturedMutatedRef(Rectangle &r) {
  return r = globalRect, capturedMutatedRef(r);
}
// CHECK10-CPP: "static int extracted(Rectangle &r) {\nreturn r = globalRect, capturedMutatedRef(r);\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK10-CPP-NEXT: "extracted(r)" [[@LINE-3]]:10 -> [[@LINE-3]]:47

#endif

int capturedArrayMutation(int x[]) {
  return (x) = 0, capturedArrayMutation(x);
}
// CHECK10-CPP: "static int extracted(int *&x) {\nreturn (x) = 0, capturedArrayMutation(x);\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK10-CPP-NEXT: "extracted(x)" [[@LINE-3]]:10 -> [[@LINE-3]]:43
// CHECK10-C: "static int extracted(int **x) {\nreturn (*x) = 0, capturedArrayMutation(*x);\n}\n\n" [[@LINE-5]]:1 -> [[@LINE-5]]:1
// CHECK10-C-NEXT: "extracted(&x)" [[@LINE-5]]:10 -> [[@LINE-5]]:43
;
void mutationOutsideOfExtractedCode(int x) {
  x = 0;
  x = 1, mutationOutsideOfExtractedCode(x + 1);
  x = 2;
  return 0;
}
// CHECK10-CPP: "static void extracted(int x) {\nmutationOutsideOfExtractedCode(x + 1);\n}\n\n" [[@LINE-6]]:1 -> [[@LINE-6]]:1
// CHECK10-CPP-NEXT: "extracted(x)" [[@LINE-5]]:10 -> [[@LINE-5]]:47

// RUN: clang-refactor-test perform -action extract -selected=%s:129:10-129:97 -selected=%s:140:10-140:47 -selected=%s:148:10-148:43 -selected=%s:157:10-157:47 %s | FileCheck --check-prefix=CHECK10-CPP %s
// RUN: clang-refactor-test perform -action extract -selected=%s:129:10-129:97 -selected=%s:148:10-148:43 %s -x c | FileCheck --check-prefix=CHECK10-C %s
;
void extractStatementRangeRewritePointerUse(int x) {
  extractStatementRangeRewritePointerUse(x);
  x = 1;
  extractStatementRangeRewritePointerUse(x);
  extractStatementRangeRewritePointerUse(x);
}
// CHECK11: "static void extracted(int *x) {\n*x = 1;\n  extractStatementRangeRewritePointerUse(*x);\n}\n\n" [[@LINE-6]]:1
// CHECK11-NEXT: "extracted(&x)" [[@LINE-5]]:3 -> [[@LINE-4]]:44

// RUN: clang-refactor-test perform -action extract -selected=%s:169:3-170:10 %s -x c | FileCheck --check-prefix=CHECK11 %s

void mutationAfterUse(int x) {
  int y = x;
  (void)y;
  y = 2;
}
// CHECK12: "static void extracted(int &y) {\n(void)y;\n  y = 2;\n}\n\n"
// RUN: clang-refactor-test perform -action extract -selected=%s:180:3-181:4 %s | FileCheck --check-prefix=CHECK12 %s

enum TagEnum {
  TagEnumA, TagEnumB
};

int tagEnumType(enum TagEnum e) {
  return tagEnumType(e);
}
// CHECK13: (enum TagEnum e)
// RUN: clang-refactor-test perform -action extract -selected=%s:191:10-191:24 %s | FileCheck --check-prefix=CHECK13 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:191:10-191:24 %s -x c | FileCheck --check-prefix=CHECK13 %s

struct TagStruct {
  int x;
};

int tagStructType(struct TagStruct *s) {
  return tagStructType(s);
}
// CHECK14: (struct TagStruct *s)
// RUN: clang-refactor-test perform -action extract -selected=%s:202:10-202:26 %s | FileCheck --check-prefix=CHECK14 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:202:10-202:26 %s -x c | FileCheck --check-prefix=CHECK14 %s

namespace {
struct Foo { int x; };
}

void anonymousNamespaceTypes() {
  Foo x;
// anonymous-ns-type1-begin: +1:1
  x.x = 0;
// anonymous-ns-type1-end: +0:1
// CHECK-ANON: "static void extracted(Foo &x) {\nx.x = 0;\n}\n\n"
// anonymous-ns-type2-begin: +1:7
  x = Foo { 0 };
// anonymous-ns-type2-end: -1:16
// CHECK-ANON: "static Foo extracted() {\nreturn Foo { 0 };\n}\n\n"
}
// RUN: clang-refactor-test perform -action extract -selected=anonymous-ns-type1 -selected=anonymous-ns-type2 %s -std=c++11 | FileCheck --check-prefix=CHECK-ANON %s
