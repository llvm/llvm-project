// Test is line- and column-sensitive; see below.

struct X {
  X(int value);
  X(const X& x);
protected:
  ~X();
private:
  operator X*();

  void constMemberFunction() const;
  template<typename T>
  void constMemberFunctionTemplate() const;

  static void staticMemberFunction();
  template<typename T>
  static void staticMemberFunctionTemplate();

  virtual void virtualMemberFunction();
  virtual void pureVirtualMemberFunction() = 0;
};

X::X(int value) {
}

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: load-classes.cpp:3:8: StructDecl=X:3:8 (Definition) Extent=[3:1 - 21:2]
// CHECK: load-classes.cpp:4:3: CXXConstructor=X:4:3 (converting constructor) Extent=[4:3 - 4:15] [access=public]
// FIXME: missing TypeRef in the constructor name
// CHECK: load-classes.cpp:4:9: ParmDecl=value:4:9 (Definition) Extent=[4:5 - 4:14]
// CHECK: load-classes.cpp:5:3: CXXConstructor=X:5:3 (copy constructor) (converting constructor) Extent=[5:3 - 5:16] [access=public]
// FIXME: missing TypeRef in the constructor name
// CHECK: load-classes.cpp:5:14: ParmDecl=x:5:14 (Definition) Extent=[5:5 - 5:15]
// CHECK: load-classes.cpp:5:11: TypeRef=struct X:3:8 Extent=[5:11 - 5:12]
// CHECK: load-classes.cpp:6:1: CXXAccessSpecifier=:6:1 (Definition) Extent=[6:1 - 6:11] [access=protected]
// CHECK: load-classes.cpp:7:3: CXXDestructor=~X:7:3 Extent=[7:3 - 7:7] [access=protected]
// FIXME: missing TypeRef in the destructor name
// CHECK: load-classes.cpp:8:1: CXXAccessSpecifier=:8:1 (Definition) Extent=[8:1 - 8:9] [access=private]
// CHECK: load-classes.cpp:9:3: CXXConversion=operator X *:9:3 Extent=[9:3 - 9:16] [access=private]
// CHECK: load-classes.cpp:9:12: TypeRef=struct X:3:8 Extent=[9:12 - 9:13]
// CHECK: load-classes.cpp:11:8: CXXMethod=constMemberFunction:11:8 (const) Extent=[11:3 - 11:35] [access=private]
// CHECK: load-classes.cpp:13:8: FunctionTemplate=constMemberFunctionTemplate:13:8 (const) Extent=[12:3 - 13:43] [access=private]
// CHECK: load-classes.cpp:12:21: TemplateTypeParameter=T:12:21 (Definition) Extent=[12:12 - 12:22] [access=public]
// CHECK: load-classes.cpp:15:15: CXXMethod=staticMemberFunction:15:15 (static) Extent=[15:3 - 15:37] [access=private]
// CHECK: load-classes.cpp:17:15: FunctionTemplate=staticMemberFunctionTemplate:17:15 (static) Extent=[16:3 - 17:45] [access=private]
// CHECK: load-classes.cpp:16:21: TemplateTypeParameter=T:16:21 (Definition) Extent=[16:12 - 16:22] [access=public]
// CHECK: load-classes.cpp:19:16: CXXMethod=virtualMemberFunction:19:16 (virtual) Extent=[19:3 - 19:39] [access=private]
// CHECK: load-classes.cpp:20:16: CXXMethod=pureVirtualMemberFunction:20:16 (virtual) (pure) Extent=[20:3 - 20:47] [access=private]
// CHECK: load-classes.cpp:23:4: CXXConstructor=X:23:4 (Definition) (converting constructor) Extent=[23:1 - 24:2] [access=public]
// CHECK: load-classes.cpp:23:1: TypeRef=struct X:3:8 Extent=[23:1 - 23:2]
// CHECK: load-classes.cpp:23:10: ParmDecl=value:23:10 (Definition) Extent=[23:6 - 23:15]
// CHECK: load-classes.cpp:23:17: CompoundStmt= Extent=[23:17 - 24:2]
