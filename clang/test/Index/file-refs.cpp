namespace NS {
  class C {
  public:
    C() { }
    void m();
  };
}

void NS::C::m() {
  C c;
  c.m();
}

void f() {
  NS::C c1();
  NS::C c2 = NS::C();
}

void over(int);
void over(float);

void test_over() {
  over(0);
  over(0.0f);
}

template <typename T>
T tf(T t) {
  return t;
}

namespace Test2 {

struct S {
  S(int x, int y);
  S();
};

typedef S Cake;

void f() {
  Cake p;
  p = Test2::S(0,2);
  p = Test2::Cake(0,2);
}

}

// RUN: c-index-test \

// RUN:  -file-refs-at=%s:9:7 \
// CHECK:      NamespaceRef=NS:1:11
// CHECK-NEXT: Namespace=NS:1:11 (Definition) =[1:11 - 1:13]
// CHECK-NEXT: NamespaceRef=NS:1:11 =[9:6 - 9:8]
// CHECK-NEXT: NamespaceRef=NS:1:11 =[15:3 - 15:5]
// CHECK-NEXT: NamespaceRef=NS:1:11 =[16:3 - 16:5]
// CHECK-NEXT: NamespaceRef=NS:1:11 =[16:14 - 16:16]

// RUN:  -file-refs-at=%s:2:9 \
// CHECK-NEXT: ClassDecl=C:2:9 (Definition)
// CHECK-NEXT: ClassDecl=C:2:9 (Definition) =[2:9 - 2:10]
// CHECK-NEXT: CXXConstructor=C:4:5 (Definition) (default constructor) =[4:5 - 4:6]
// CHECK-NEXT: TypeRef=class NS::C:2:9 =[9:10 - 9:11]
// CHECK-NEXT: TypeRef=class NS::C:2:9 =[10:3 - 10:4]
// CHECK-NEXT: TypeRef=class NS::C:2:9 =[15:7 - 15:8]
// CHECK-NEXT: TypeRef=class NS::C:2:9 =[16:7 - 16:8]
// CHECK-NEXT: TypeRef=class NS::C:2:9 =[16:18 - 16:19]

// RUN:  -file-refs-at=%s:16:18 \
// CHECK-NEXT: CallExpr=C:4:5
// CHECK-NEXT: ClassDecl=C:2:9 (Definition) =[2:9 - 2:10]
// CHECK-NEXT: CXXConstructor=C:4:5 (Definition) (default constructor) =[4:5 - 4:6]
// CHECK-NEXT: TypeRef=class NS::C:2:9 =[9:10 - 9:11]
// CHECK-NEXT: TypeRef=class NS::C:2:9 =[10:3 - 10:4]
// CHECK-NEXT: TypeRef=class NS::C:2:9 =[15:7 - 15:8]
// CHECK-NEXT: TypeRef=class NS::C:2:9 =[16:7 - 16:8]
// CHECK-NEXT: TypeRef=class NS::C:2:9 =[16:18 - 16:19]

// RUN:  -file-refs-at=%s:20:8 \
// CHECK-NEXT: FunctionDecl=over:20:6
// CHECK-NEXT: FunctionDecl=over:20:6 =[20:6 - 20:10]
// CHECK-NEXT: DeclRefExpr=over:20:6 =[24:3 - 24:7]

// RUN:  -file-refs-at=%s:28:1 \
// CHECK-NEXT: TypeRef=T:27:20
// FIXME: Missing TemplateTypeParameter=T:27:20 (Definition)
// CHECK-NEXT: TypeRef=T:27:20 =[28:1 - 28:2]
// CHECK-NEXT: TypeRef=T:27:20 =[28:6 - 28:7]

// RUN:  -file-refs-at=%s:43:14 \
// CHECK-NEXT: CallExpr=S:35:3
// CHECK-NEXT: StructDecl=S:34:8 (Definition) =[34:8 - 34:9]
// CHECK-NEXT: CXXConstructor=S:35:3 =[35:3 - 35:4]
// CHECK-NEXT: CXXConstructor=S:36:3 (default constructor) =[36:3 - 36:4]
// CHECK-NEXT: TypeRef=struct Test2::S:34:8 =[39:9 - 39:10]
// CHECK-NEXT: TypeRef=struct Test2::S:34:8 =[43:14 - 43:15]

// RUN:  -file-refs-at=%s:44:16 \
// CHECK-NEXT: CallExpr=S:35:3
// CHECK-NEXT: TypedefDecl=Cake:39:11 (Definition) =[39:11 - 39:15]
// CHECK-NEXT: TypeRef=Test2::Cake:39:11 =[42:3 - 42:7]
// CHECK-NEXT: TypeRef=Test2::Cake:39:11 =[44:14 - 44:18]

// RUN:   %s | FileCheck %s
