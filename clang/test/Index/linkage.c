// RUN: c-index-test -test-print-linkage-source %s | FileCheck %s

enum Baz { Qux = 0 };
int x;
void foo();
static int w;
void bar(int y) {
  static int z;
  int k;
}
extern int n;
static int wibble(int);

void ena(int (*dio)(int tria));

static int test2;
void f16(void) {
  extern int test2;
}


// CHECK: EnumDecl=Baz:3:6 (Definition)linkage=External
// CHECK: EnumConstantDecl=Qux:3:12 (Definition)linkage=External
// CHECK: VarDecl=x:4:5linkage=External
// CHECK: FunctionDecl=foo:5:6linkage=External
// CHECK: VarDecl=w:6:12linkage=Internal
// CHECK: FunctionDecl=bar:7:6 (Definition)linkage=External
// CHECK: ParmDecl=y:7:14 (Definition)linkage=NoLinkage
// CHECK: VarDecl=z:8:14 (Definition)linkage=NoLinkage
// CHECK: VarDecl=k:9:7 (Definition)linkage=NoLinkage
// CHECK: VarDecl=n:11:12linkage=External
// CHECK: FunctionDecl=wibble:12:12linkage=Internal
// CHECK: ParmDecl=:12:22 (Definition)linkage=NoLinkage
// CHECK: FunctionDecl=ena:14:6linkage=External
// CHECK: ParmDecl=dio:14:16 (Definition)linkage=NoLinkage
// CHECK: ParmDecl=tria:14:25 (Definition)linkage=NoLinkage
// CHECK: VarDecl=test2{{.*}}linkage=Internal
// CHECK: VarDecl=test2{{.*}}linkage=Internal
