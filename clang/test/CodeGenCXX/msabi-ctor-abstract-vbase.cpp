// RUN: %clang_cc1 -triple x86_64-windows-msvc %s -emit-llvm -fexceptions -o - | FileCheck %s

// PR41065: As background, when constructing a complete object, virtual bases
// are constructed first. If an exception is thrown while constructing a
// subobject later, those virtual bases are destroyed, so sema references the
// relevant constructors and destructors of every base class in case they are
// needed.
//
// However, an abstract class can never be used to construct a complete object.
// In the Itanium C++ ABI, this works out nicely, because we never end up
// emitting the "complete" constructor variant, only the "base" constructor
// variant, which can be called by constructors of derived classes. For various
// reasons, Sema does not mark ctors and dtors of virtual bases referenced when
// the constructor of an abstract class is emitted.
//
// In the Microsoft ABI, there are no complete/base variants, so before PR41065
// was fixed, the constructor of an abstract class could reference special
// members of a virtual base without marking them referenced. This could lead to
// unresolved symbol errors at link time.
//
// The fix is to implement the same optimization as Sema: If the class is
// abstract, don't bother initializing its virtual bases. The "is this class the
// most derived class" check in the constructor will never pass, and the virtual
// base constructor calls are always dead. Skip them.

struct A {
  A();
  virtual void f() = 0;
  virtual ~A();
};

// B has an implicit inline dtor, but is still abstract.
struct B : A {
  B(int n);
  int n;
};

// Still abstract
struct C : virtual B {
  C(int n);
  //void f() override;
};

// Not abstract, D::D calls C::C and B::B.
struct D : C {
  D(int n);
  void f() override;
};

void may_throw();
C::C(int n) : B(n) { may_throw(); }

// No branches, no constructor calls before may_throw();
//
// CHECK-LABEL: define dso_local noundef %struct.C* @"??0C@@QEAA@H@Z"(%struct.C* {{[^,]*}} returned align 8 dereferenceable(8) %this, i32 noundef %n, i32 noundef %is_most_derived)
// CHECK-NOT: br i1
// CHECK-NOT: {{call.*@"\?0}}
// CHECK: call void @"?may_throw@@YAXXZ"()
// no cleanups


D::D(int n) : C(n), B(n) { may_throw(); }

// Conditionally construct (and destroy) vbase B, unconditionally C.
//
// CHECK-LABEL: define dso_local noundef %struct.D* @"??0D@@QEAA@H@Z"(%struct.D* {{[^,]*}} returned align 8 dereferenceable(8) %this, i32 noundef %n, i32 noundef %is_most_derived)
// CHECK: icmp ne i32 {{.*}}, 0
// CHECK: br i1
// CHECK: call noundef %struct.B* @"??0B@@QEAA@H@Z"
// CHECK: br label
// CHECK: invoke noundef %struct.C* @"??0C@@QEAA@H@Z"
// CHECK: invoke void @"?may_throw@@YAXXZ"()
// CHECK: cleanuppad
// CHECK: call void @"??1C@@UEAA@XZ"
// CHECK: cleanupret
//
// CHECK: cleanuppad
// CHECK: icmp ne i32 {{.*}}, 0
// CHECK: br i1
// CHECK: call void @"??1B@@UEAA@XZ"
// CHECK: br label
// CHECK: cleanupret
