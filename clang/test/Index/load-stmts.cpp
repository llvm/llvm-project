typedef int T;
struct X { int a, b; };
void f(int x) {
  for (T y = x; T z = x; ++x) {
  }
  if (T *z2 = &x) { }
  while (T *z3 = &x) { }
  switch (T z4 = x) {
  case 17: break;
  }
}

// Test handling of C++ base specifiers.
class A {
  void doA();
};

class B {
  void doB();
};

class C : public A, private B {
  void doC();
};

class D : virtual public C, virtual private A {};

namespace std {
  class type_info { };
}

void test_exprs(C *c) {
  int typeid_marker;
  typeid(C);
  typeid(c);
  typedef int Integer;
  Integer *int_ptr;
  int_ptr->Integer::~Integer();
}

namespace N {
  int f(int);
  float f(float);

  template<typename T> T g(T);
  template<typename T> T g(T*);
}

template<typename T>
void test_dependent_exprs(T t) {
  N::f(t);
  typedef T type;
  N::g<type>(t);
  type::template f<type*>(t);
  t->type::template f<type*>();
}

struct Y {
  int f(int);
  float f(float);

  template<typename T> T g(T);
  template<typename T> T g(T*);
};

template<typename T>
void test_more_dependent_exprs(T t, Y y) {
  y.Y::f(t);
  typedef T type;
  y.g<type>(t);
}

struct Pair {
  Pair(int, int);
};

void *operator new(__SIZE_TYPE__, void*) throw();

void test_more_exprs(void *mem, int i, int j) {
  new (mem) Pair(i, j);
  typedef int Integer;
  (void)Integer(i);
  (Integer)i;
  Integer();
}

template<typename T>
void test_even_more_dependent_exprs(T t, Y y) {
  typedef T type;
  (void)type(t, y);
  (void)__has_nothrow_assign(type);
}

struct Base {
  Base(int);
};

struct Derived : public Base {
  Derived(int x);
  int member;
};

Derived::Derived(int x) 
  : member(x), Base(x) {
}

void considered_harmful(int x) {
 start_over:
  void *ptr = &&start_over;
  if (x > 17)
    goto *ptr;
  else
    goto start_over;
}

void casts(int *ip) {
  (void)reinterpret_cast<float *>(ip);
}

// RUN: c-index-test -test-load-source all -fno-delayed-template-parsing -frtti %s | FileCheck %s
// CHECK: load-stmts.cpp:1:13: TypedefDecl=T:1:13 (Definition) Extent=[1:1 - 1:14]
// CHECK: load-stmts.cpp:2:8: StructDecl=X:2:8 (Definition) Extent=[2:1 - 2:23]
// CHECK: load-stmts.cpp:2:16: FieldDecl=a:2:16 (Definition) Extent=[2:12 - 2:17]
// CHECK: load-stmts.cpp:2:19: FieldDecl=b:2:19 (Definition) Extent=[2:12 - 2:20]
// CHECK: load-stmts.cpp:3:6: FunctionDecl=f:3:6 (Definition) Extent=[3:1 - 11:2]
// CHECK: load-stmts.cpp:3:12: ParmDecl=x:3:12 (Definition) Extent=[3:8 - 3:13]
// CHECK: load-stmts.cpp:4:10: VarDecl=y:4:10 (Definition) Extent=[4:8 - 4:15]
// CHECK: load-stmts.cpp:4:8: TypeRef=T:1:13 Extent=[4:8 - 4:9]
// CHECK: load-stmts.cpp:4:14: DeclRefExpr=x:3:12 Extent=[4:14 - 4:15]
// CHECK: load-stmts.cpp:4:19: VarDecl=z:4:19 (Definition) Extent=[4:17 - 4:24]
// CHECK: load-stmts.cpp:4:17: TypeRef=T:1:13 Extent=[4:17 - 4:18]
// CHECK: load-stmts.cpp:4:23: DeclRefExpr=x:3:12 Extent=[4:23 - 4:24]
// CHECK: load-stmts.cpp:4:19: UnexposedExpr=z:4:19 Extent=[4:19 - 4:20]
// CHECK: load-stmts.cpp:4:19: DeclRefExpr=z:4:19 Extent=[4:19 - 4:20]
// CHECK: load-stmts.cpp:4:26: UnaryOperator= Extent=[4:26 - 4:29]
// CHECK: load-stmts.cpp:4:28: DeclRefExpr=x:3:12 Extent=[4:28 - 4:29]
// CHECK: load-stmts.cpp:6:10: VarDecl=z2:6:10 (Definition) Extent=[6:7 - 6:17]
// CHECK: load-stmts.cpp:6:7: TypeRef=T:1:13 Extent=[6:7 - 6:8]
// CHECK: load-stmts.cpp:6:15: UnaryOperator= Extent=[6:15 - 6:17]
// CHECK: load-stmts.cpp:6:16: DeclRefExpr=x:3:12 Extent=[6:16 - 6:17]
// CHECK: load-stmts.cpp:6:10: UnexposedExpr=z2:6:10 Extent=[6:10 - 6:12]
// CHECK: load-stmts.cpp:6:10: DeclRefExpr=z2:6:10 Extent=[6:10 - 6:12]
// CHECK: load-stmts.cpp:7:13: VarDecl=z3:7:13 (Definition) Extent=[7:10 - 7:20]
// CHECK: load-stmts.cpp:7:10: TypeRef=T:1:13 Extent=[7:10 - 7:11]
// CHECK: load-stmts.cpp:7:18: UnaryOperator= Extent=[7:18 - 7:20]
// CHECK: load-stmts.cpp:7:19: DeclRefExpr=x:3:12 Extent=[7:19 - 7:20]
// CHECK: load-stmts.cpp:7:13: UnexposedExpr=z3:7:13 Extent=[7:13 - 7:15]
// CHECK: load-stmts.cpp:7:13: DeclRefExpr=z3:7:13 Extent=[7:13 - 7:15]
// CHECK: load-stmts.cpp:8:13: VarDecl=z4:8:13 (Definition) Extent=[8:11 - 8:19]
// CHECK: load-stmts.cpp:8:11: TypeRef=T:1:13 Extent=[8:11 - 8:12]
// CHECK: load-stmts.cpp:8:18: DeclRefExpr=x:3:12 Extent=[8:18 - 8:19]
// CHECK: load-stmts.cpp:8:13: DeclRefExpr=z4:8:13 Extent=[8:13 - 8:15]
// CHECK: load-stmts.cpp:9:8: IntegerLiteral= Extent=[9:8 - 9:10]
// CHECK: load-stmts.cpp:14:7: ClassDecl=A:14:7 (Definition) Extent=[14:1 - 16:2]
// CHECK: load-stmts.cpp:15:8: CXXMethod=doA:15:8 Extent=[15:3 - 15:13]
// CHECK: load-stmts.cpp:18:7: ClassDecl=B:18:7 (Definition) Extent=[18:1 - 20:2]
// CHECK: load-stmts.cpp:19:8: CXXMethod=doB:19:8 Extent=[19:3 - 19:13]
// CHECK: load-stmts.cpp:22:7: ClassDecl=C:22:7 (Definition) Extent=[22:1 - 24:2]
// CHECK: load-stmts.cpp:22:18: C++ base class specifier=A:14:7 [access=public isVirtual=false]
// CHECK: load-stmts.cpp:22:29: C++ base class specifier=B:18:7 [access=private isVirtual=false]
// CHECK: load-stmts.cpp:23:8: CXXMethod=doC:23:8 Extent=[23:3 - 23:13]
// CHECK: load-stmts.cpp:26:7: ClassDecl=D:26:7 (Definition) Extent=[26:1 - 26:49]
// CHECK: load-stmts.cpp:26:26: C++ base class specifier=C:22:7 [access=public isVirtual=true]
// CHECK: load-stmts.cpp:26:45: C++ base class specifier=A:14:7 [access=private isVirtual=true]
// CHECK: load-stmts.cpp:33:7: VarDecl=typeid_marker:33:7 (Definition)
// CHECK: load-stmts.cpp:34:10: TypeRef=class C:22:7 Extent=[34:10 - 34:11]
// CHECK: load-stmts.cpp:35:10: DeclRefExpr=c:32:20 Extent=[35:10 - 35:11]
// CHECK: load-stmts.cpp:37:12: VarDecl=int_ptr:37:12 (Definition) Extent=[37:3 - 37:19]
// CHECK: load-stmts.cpp:37:3: TypeRef=Integer:36:15 Extent=[37:3 - 37:10]
// CHECK: load-stmts.cpp:38:3: DeclRefExpr=int_ptr:37:12 Extent=[38:3 - 38:10]
// CHECK: load-stmts.cpp:38:12: TypeRef=Integer:36:15 Extent=[38:12 - 38:19]
// CHECK: load-stmts.cpp:38:22: TypeRef=Integer:36:15 Extent=[38:22 - 38:29]
// CHECK: load-stmts.cpp:50:6: FunctionTemplate=test_dependent_exprs:50:6 (Definition)
// CHECK: load-stmts.cpp:51:3: CallExpr= Extent=[51:3 - 51:10]
// CHECK: load-stmts.cpp:51:3: NamespaceRef=N:41:11 Extent=[51:3 - 51:4]
// CHECK: load-stmts.cpp:51:8: DeclRefExpr=t:50:29 Extent=[51:8 - 51:9]
// CHECK: load-stmts.cpp:52:13: TypedefDecl=type:52:13 (Definition) Extent=[52:3 - 52:17]
// CHECK: load-stmts.cpp:53:3: CallExpr= Extent=[53:3 - 53:16]
// CHECK: load-stmts.cpp:53:3: NamespaceRef=N:41:11 Extent=[53:3 - 53:4]
// CHECK: load-stmts.cpp:53:8: TypeRef=type:52:13 Extent=[53:8 - 53:12]
// CHECK: load-stmts.cpp:53:14: DeclRefExpr=t:50:29 Extent=[53:14 - 53:15]
// CHECK: load-stmts.cpp:54:3: CallExpr= Extent=[54:3 - 54:29]
// CHECK: load-stmts.cpp:54:3: TypeRef=type:52:13 Extent=[54:3 - 54:7]
// CHECK: load-stmts.cpp:54:20: TypeRef=type:52:13 Extent=[54:20 - 54:24]
// CHECK: load-stmts.cpp:54:27: DeclRefExpr=t:50:29 Extent=[54:27 - 54:28]
// CHECK: load-stmts.cpp:55:3: CallExpr= Extent=[55:3 - 55:31]
// CHECK: load-stmts.cpp:55:3: DeclRefExpr=t:50:29 Extent=[55:3 - 55:4]
// CHECK: load-stmts.cpp:55:23: TypeRef=type:52:13 Extent=[55:23 - 55:27]
// CHECK: load-stmts.cpp:67:6: FunctionTemplate=test_more_dependent_exprs:67:6 (Definition)
// CHECK: load-stmts.cpp:68:3: CallExpr= Extent=[68:3 - 68:12]
// CHECK: load-stmts.cpp:68:3: DeclRefExpr=y:67:39 Extent=[68:3 - 68:4]
// CHECK: load-stmts.cpp:68:5: TypeRef=struct Y:58:8 Extent=[68:5 - 68:6]
// CHECK: load-stmts.cpp:68:10: DeclRefExpr=t:67:34 Extent=[68:10 - 68:11]
// CHECK: load-stmts.cpp:70:3: CallExpr= Extent=[70:3 - 70:15]
// CHECK: load-stmts.cpp:70:3: DeclRefExpr=y:67:39 Extent=[70:3 - 70:4]
// CHECK: load-stmts.cpp:70:7: TypeRef=type:69:13 Extent=[70:7 - 70:11]
// CHECK: load-stmts.cpp:70:13: DeclRefExpr=t:67:34 Extent=[70:13 - 70:14]
// CHECK: load-stmts.cpp:79:6: FunctionDecl=test_more_exprs:79:6 (Definition)
// CHECK: load-stmts.cpp:80:8: DeclRefExpr=mem:79:28 Extent=[80:8 - 80:11]
// CHECK: load-stmts.cpp:80:13: TypeRef=struct Pair:73:8 Extent=[80:13 - 80:17]
// CHECK: load-stmts.cpp:80:18: DeclRefExpr=i:79:37 Extent=[80:18 - 80:19]
// CHECK: load-stmts.cpp:80:21: DeclRefExpr=j:79:44 Extent=[80:21 - 80:22]
// CHECK: load-stmts.cpp:82:9: TypeRef=Integer:81:15 Extent=[82:9 - 82:16]
// CHECK: load-stmts.cpp:82:17: DeclRefExpr=i:79:37 Extent=[82:17 - 82:18]
// CHECK: load-stmts.cpp:83:3: CStyleCastExpr= Extent=[83:3 - 83:13]
// CHECK: load-stmts.cpp:83:4: TypeRef=Integer:81:15 Extent=[83:4 - 83:11]
// CHECK: load-stmts.cpp:83:12: UnexposedExpr=i:79:37 Extent=[83:12 - 83:13]
// CHECK: load-stmts.cpp:83:12: DeclRefExpr=i:79:37 Extent=[83:12 - 83:13]
// CHECK: load-stmts.cpp:84:3: UnexposedExpr= Extent=[84:3 - 84:12]
// CHECK: load-stmts.cpp:84:3: TypeRef=Integer:81:15 Extent=[84:3 - 84:10]
// CHECK: load-stmts.cpp:90:9: TypeRef=type:89:13 Extent=[90:9 - 90:13]
// CHECK: load-stmts.cpp:90:14: DeclRefExpr=t:88:39 Extent=[90:14 - 90:15]
// CHECK: load-stmts.cpp:90:17: DeclRefExpr=y:88:44 Extent=[90:17 - 90:18]
// CHECK: load-stmts.cpp:91:9: UnexposedExpr= Extent=[91:9 - 91:35]
// CHECK: load-stmts.cpp:91:30: TypeRef=type:89:13 Extent=[91:30 - 91:34]
// CHECK: load-stmts.cpp:103:10: CXXConstructor=Derived:103:10 (Definition)
// CHECK: load-stmts.cpp:103:1: TypeRef=struct Derived:98:8 Extent=[103:1 - 103:
// FIXME: Missing TypeRef for constructor name.
// CHECK: load-stmts.cpp:103:22: ParmDecl=x:103:22 (Definition) 
// CHECK: load-stmts.cpp:104:5: MemberRef=member:100:7 Extent=[104:5 - 104:11]
// CHECK: load-stmts.cpp:104:12: DeclRefExpr=x:103:22 Extent=[104:12 - 104:13]
// CHECK: load-stmts.cpp:104:16: TypeRef=struct Base:94:8 Extent=[104:16 - 104:2
// CHECK: load-stmts.cpp:104:16: CallExpr=Base:95:3 Extent=[104:16 - 104:23]
// CHECK: load-stmts.cpp:104:21: DeclRefExpr=x:103:22 Extent=[104:21 - 104:22]
// CHECK: load-stmts.cpp:107:6: FunctionDecl=considered_harmful:107:6 (Definition)
// CHECK: load-stmts.cpp:108:2: LabelStmt=start_over Extent=[108:2 - 109:28]
// CHECK: load-stmts.cpp:109:17: LabelRef=start_over:108:2 Extent=[109:17 - 109:27]
// CHECK: load-stmts.cpp:113:10: LabelRef=start_over:108:2 Extent=[113:10 - 113:20]
// CHECK: load-stmts.cpp:117:35: UnexposedExpr=ip:116:17 Extent=[117:35 - 117:37]
// CHECK: load-stmts.cpp:117:35: DeclRefExpr=ip:116:17 Extent=[117:35 - 117:37]

