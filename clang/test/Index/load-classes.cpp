// Test is line- and column-sensitive; see below.

struct X {
  X(int value);
  X(const X& x);
  ~X();
  operator X*();
};

X::X(int value) {
}

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: load-classes.cpp:3:8: StructDecl=X:3:8 (Definition) Extent=[3:1 - 8:2]
// CHECK: load-classes.cpp:4:3: CXXConstructor=X:4:3 Extent=[4:3 - 4:15]
// FIXME: missing TypeRef in the constructor name
// CHECK: load-classes.cpp:4:9: ParmDecl=value:4:9 (Definition) Extent=[4:5 - 4:14]
// CHECK: load-classes.cpp:5:3: CXXConstructor=X:5:3 Extent=[5:3 - 5:16]
// FIXME: missing TypeRef in the constructor name
// CHECK: load-classes.cpp:5:14: ParmDecl=x:5:14 (Definition) Extent=[5:5 - 5:15]
// CHECK: load-classes.cpp:5:11: TypeRef=struct X:3:8 Extent=[5:11 - 5:12]
// CHECK: load-classes.cpp:6:3: CXXDestructor=~X:6:3 Extent=[6:3 - 6:7]
// FIXME: missing TypeRef in the destructor name
// CHECK: load-classes.cpp:7:3: CXXConversion=operator struct X *:7:3 Extent=[7:3 - 7:16]
// CHECK: load-classes.cpp:7:12: TypeRef=struct X:3:8 Extent=[7:12 - 7:13]
// CHECK: load-classes.cpp:10:4: CXXConstructor=X:10:4 (Definition) Extent=[10:1 - 11:2]
// CHECK: load-classes.cpp:10:1: TypeRef=struct X:3:8 Extent=[10:1 - 10:2]
// CHECK: load-classes.cpp:10:10: ParmDecl=value:10:10 (Definition) Extent=[10:6 - 10:15]
