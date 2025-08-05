// RUN: %clang_cc1 -no-enable-noundef-analysis -emit-llvm -fno-rtti %s -std=c++11 -o - -mconstructor-aliases -triple=i386-pc-win32 -fno-rtti > %t
// RUN: FileCheck %s < %t
// vftables are emitted very late, so do another pass to try to keep the checks
// in source order.
// RUN: FileCheck --check-prefix DTORS %s < %t
// RUN: FileCheck --check-prefix DTORS2 %s < %t
// RUN: FileCheck --check-prefix DTORS3 %s < %t
// RUN: FileCheck --check-prefix DTORS4 %s < %t
//
// RUN: %clang_cc1 -emit-llvm %s -o - -mconstructor-aliases -triple=x86_64-pc-win32 -fno-rtti -std=c++11 | FileCheck --check-prefix DTORS-X64 %s

namespace basic {

class A {
 public:
  A() { }
  ~A();
};

void no_constructor_destructor_infinite_recursion() {
  A a;

// CHECK:      define linkonce_odr dso_local x86_thiscallcc ptr @"??0A@basic@@QAE@XZ"(ptr {{[^,]*}} returned {{[^,]*}} %this) {{.*}} comdat {{.*}} {
// CHECK:        [[THIS_ADDR:%[.0-9A-Z_a-z]+]] = alloca ptr, align 4
// CHECK-NEXT:   store ptr %this, ptr [[THIS_ADDR]], align 4
// CHECK-NEXT:   [[T1:%[.0-9A-Z_a-z]+]] = load ptr, ptr [[THIS_ADDR]]
// CHECK-NEXT:   ret ptr [[T1]]
// CHECK-NEXT: }
}

A::~A() {
// Make sure that the destructor doesn't call itself:
// CHECK: define {{.*}} @"??1A@basic@@QAE@XZ"
// CHECK-NOT: call void @"??1A@basic@@QAE@XZ"
// CHECK: ret
}

struct B {
  B();
};

// Tests that we can define constructors outside the class (PR12784).
B::B() {
  // CHECK: define dso_local x86_thiscallcc ptr @"??0B@basic@@QAE@XZ"(ptr {{[^,]*}} returned {{[^,]*}} %this)
  // CHECK: ret
}

struct C {
  virtual ~C() {
// DTORS:      define linkonce_odr dso_local x86_thiscallcc ptr @"??_GC@basic@@UAEPAXI@Z"(ptr {{[^,]*}} %this, i32 %should_call_delete) {{.*}} comdat {{.*}} {
// DTORS:        store i32 %should_call_delete, ptr %[[SHOULD_DELETE_VAR:[0-9a-z._]+]], align 4
// DTORS:        store ptr %{{.*}}, ptr %[[RETVAL:retval]]
// DTORS:        %[[SHOULD_DELETE_VALUE:[0-9a-z._]+]] = load i32, ptr %[[SHOULD_DELETE_VAR]]
// DTORS:        call x86_thiscallcc void @"??1C@basic@@UAE@XZ"(ptr {{[^,]*}} %[[THIS:[0-9a-z]+]])
// DTORS-NEXT:   %[[CONDITION:[0-9]+]] = icmp eq i32 %[[SHOULD_DELETE_VALUE]], 0
// DTORS-NEXT:   br i1 %[[CONDITION]], label %[[CONTINUE_LABEL:[0-9a-z._]+]], label %[[CALL_DELETE_LABEL:[0-9a-z._]+]]
//
// DTORS:      [[CALL_DELETE_LABEL]]
// DTORS-NEXT:   call void @"??3@YAXPAX@Z"(ptr %[[THIS]])
// DTORS-NEXT:   br label %[[CONTINUE_LABEL]]
//
// DTORS:      [[CONTINUE_LABEL]]
// DTORS-NEXT:   %[[RET:.*]] = load ptr, ptr %[[RETVAL]]
// DTORS-NEXT:   ret ptr %[[RET]]

// Check that we do the mangling correctly on x64.
// DTORS-X64:  @"??_GC@basic@@UEAAPEAXI@Z"
  }
  virtual void foo();
};

// Emits the vftable in the output.
void C::foo() {}

void check_vftable_offset() {
  C c;
// The vftable pointer should point at the beginning of the vftable.
// CHECK: store ptr @"??_7C@basic@@6B@", ptr {{.*}}
}

void call_complete_dtor(C *obj_ptr) {
// CHECK: define dso_local void @"?call_complete_dtor@basic@@YAXPAUC@1@@Z"(ptr %obj_ptr)
  obj_ptr->~C();
// CHECK: %[[OBJ_PTR_VALUE:.*]] = load ptr, ptr %{{.*}}, align 4
// CHECK-NEXT: %[[VTABLE:.*]] = load ptr, ptr %[[OBJ_PTR_VALUE]]
// CHECK-NEXT: %[[PVDTOR:.*]] = getelementptr inbounds ptr, ptr %[[VTABLE]], i64 0
// CHECK-NEXT: %[[VDTOR:.*]] = load ptr, ptr %[[PVDTOR]]
// CHECK-NEXT: call x86_thiscallcc ptr %[[VDTOR]](ptr {{[^,]*}} %[[OBJ_PTR_VALUE]], i32 0)
// CHECK-NEXT: ret void
}

void call_deleting_dtor(C *obj_ptr) {
// CHECK: define dso_local void @"?call_deleting_dtor@basic@@YAXPAUC@1@@Z"(ptr %obj_ptr)
  delete obj_ptr;
// CHECK:      %[[OBJ_PTR_VALUE:.*]] = load ptr, ptr %{{.*}}, align 4
// CHECK:      br i1 {{.*}}, label %[[DELETE_NULL:.*]], label %[[DELETE_NOTNULL:.*]]

// CHECK:      [[DELETE_NOTNULL]]
// CHECK-NEXT:   %[[VTABLE:.*]] = load ptr, ptr %[[OBJ_PTR_VALUE]]
// CHECK-NEXT:   %[[PVDTOR:.*]] = getelementptr inbounds ptr, ptr %[[VTABLE]], i64 0
// CHECK-NEXT:   %[[VDTOR:.*]] = load ptr, ptr %[[PVDTOR]]
// CHECK-NEXT:   call x86_thiscallcc ptr %[[VDTOR]](ptr {{[^,]*}} %[[OBJ_PTR_VALUE]], i32 1)
// CHECK:      ret void
}

void call_deleting_dtor_and_global_delete(C *obj_ptr) {
// CHECK: define dso_local void @"?call_deleting_dtor_and_global_delete@basic@@YAXPAUC@1@@Z"(ptr %obj_ptr)
  ::delete obj_ptr;
// CHECK:      %[[OBJ_PTR_VALUE:.*]] = load ptr, ptr %{{.*}}, align 4
// CHECK:      br i1 {{.*}}, label %[[DELETE_NULL:.*]], label %[[DELETE_NOTNULL:.*]]

// CHECK:      [[DELETE_NOTNULL]]
// CHECK-NEXT:   %[[VTABLE:.*]] = load ptr, ptr %[[OBJ_PTR_VALUE]]
// CHECK-NEXT:   %[[PVDTOR:.*]] = getelementptr inbounds ptr, ptr %[[VTABLE]], i64 0
// CHECK-NEXT:   %[[VDTOR:.*]] = load ptr, ptr %[[PVDTOR]]
// CHECK-NEXT:   %[[CALL:.*]] = call x86_thiscallcc ptr %[[VDTOR]](ptr {{[^,]*}} %[[OBJ_PTR_VALUE]], i32 0)
// CHECK-NEXT:   call void @"??3@YAXPAX@Z"(ptr %[[CALL]])
// CHECK:      ret void
}

struct D {
  static int foo();

  D() {
    static int ctor_static = foo();
    // CHECK that the static in the ctor gets mangled correctly:
    // CHECK: @"?ctor_static@?1???0D@basic@@QAE@XZ@4HA"
  }
  ~D() {
    static int dtor_static = foo();
    // CHECK that the static in the dtor gets mangled correctly:
    // CHECK: @"?dtor_static@?1???1D@basic@@QAE@XZ@4HA"
  }
};

void use_D() { D c; }

} // end namespace basic

namespace dtor_in_second_nvbase {

struct A {
  virtual void f();  // A needs vftable to be primary.
};
struct B {
  virtual ~B();
};
struct C : A, B {
  virtual ~C();
};

C::~C() {
// CHECK-LABEL: define dso_local x86_thiscallcc void @"??1C@dtor_in_second_nvbase@@UAE@XZ"(ptr{{[^,]*}} %this)
//      No this adjustment!
// CHECK-NOT: getelementptr
// CHECK:   load ptr, ptr %{{.*}}
//      Now we this-adjust before calling ~B.
// CHECK:   getelementptr inbounds i8, ptr %{{.*}}, i32 4
// CHECK:   call x86_thiscallcc void @"??1B@dtor_in_second_nvbase@@UAE@XZ"(ptr{{[^,]*}} %{{.*}})
// CHECK:   ret void
}

void foo() {
  C c;
}
// DTORS2-LABEL: define linkonce_odr dso_local x86_thiscallcc ptr @"??_EC@dtor_in_second_nvbase@@W3AEPAXI@Z"(ptr %this, i32 %should_call_delete)
//      Do an adjustment from B* to C*.
// DTORS2:   getelementptr i8, ptr %{{.*}}, i32 -4
// DTORS2:   %[[CALL:.*]] = tail call x86_thiscallcc ptr @"??_GC@dtor_in_second_nvbase@@UAEPAXI@Z"
// DTORS2:   ret ptr %[[CALL]]
}

namespace test2 {
// Just like dtor_in_second_nvbase, except put that in a vbase of a diamond.

// C's dtor is in the non-primary base.
struct A { virtual void f(); };
struct B { virtual ~B(); };
struct C : A, B { virtual ~C(); int c; };

// Diamond hierarchy, with C as the shared vbase.
struct D : virtual C { int d; };
struct E : virtual C { int e; };
struct F : D, E { ~F(); int f; };

F::~F() {
// CHECK-LABEL: define dso_local x86_thiscallcc void @"??1F@test2@@UAE@XZ"(ptr{{[^,]*}})
//      Do an adjustment from C vbase subobject to F as though F was the
//      complete type.
// CHECK:   getelementptr inbounds i8, ptr %{{.*}}, i32 -20
// CHECK:   store ptr
}

void foo() {
  F f;
}
// DTORS3-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"??_DF@test2@@QAEXXZ"({{.*}} {{.*}} comdat
//      Do an adjustment from C* to F*.
// DTORS3:   getelementptr i8, ptr %{{.*}}, i32 20
// DTORS3:   call x86_thiscallcc void @"??1F@test2@@UAE@XZ"
// DTORS3:   ret void

}

namespace constructors {

struct A {
  A() {}
};

struct B : A {
  B();
  ~B();
};

B::B() {
  // CHECK: define dso_local x86_thiscallcc ptr @"??0B@constructors@@QAE@XZ"(ptr {{[^,]*}} returned {{[^,]*}} %this)
  // CHECK: call x86_thiscallcc ptr @"??0A@constructors@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
  // CHECK: ret
}

struct C : virtual A {
  C();
};

C::C() {
  // CHECK: define dso_local x86_thiscallcc ptr @"??0C@constructors@@QAE@XZ"(ptr {{[^,]*}} returned {{[^,]*}} %this, i32 %is_most_derived)
  // TODO: make sure this works in the Release build too;
  // CHECK: store i32 %is_most_derived, ptr %[[IS_MOST_DERIVED_VAR:.*]], align 4
  // CHECK: %[[IS_MOST_DERIVED_VAL:.*]] = load i32, ptr %[[IS_MOST_DERIVED_VAR]]
  // CHECK: %[[SHOULD_CALL_VBASE_CTORS:.*]] = icmp ne i32 %[[IS_MOST_DERIVED_VAL]], 0
  // CHECK: br i1 %[[SHOULD_CALL_VBASE_CTORS]], label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]
  //
  // CHECK: [[INIT_VBASES]]
  // CHECK-NEXT: %[[vbptr_off:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 0
  // CHECK-NEXT: store ptr @"??_8C@constructors@@7B@", ptr %[[vbptr_off]]
  // CHECK-NEXT: getelementptr inbounds i8, ptr %{{.*}}, i32 4
  // CHECK-NEXT: call x86_thiscallcc ptr @"??0A@constructors@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
  // CHECK-NEXT: br label %[[SKIP_VBASES]]
  //
  // CHECK: [[SKIP_VBASES]]
  // Class C does not define or override methods, so shouldn't change the vfptr.
  // CHECK-NOT: @"??_7C@constructors@@6B@"
  // CHECK: ret
}

void create_C() {
  C c;
  // CHECK: define dso_local void @"?create_C@constructors@@YAXXZ"()
  // CHECK: call x86_thiscallcc ptr @"??0C@constructors@@QAE@XZ"(ptr {{[^,]*}} %c, i32 1)
  // CHECK: ret
}

struct D : C {
  D();
};

D::D() {
  // CHECK: define dso_local x86_thiscallcc ptr @"??0D@constructors@@QAE@XZ"(ptr {{[^,]*}} returned {{[^,]*}} %this, i32 %is_most_derived) unnamed_addr
  // CHECK: store i32 %is_most_derived, ptr %[[IS_MOST_DERIVED_VAR:.*]], align 4
  // CHECK: %[[IS_MOST_DERIVED_VAL:.*]] = load i32, ptr %[[IS_MOST_DERIVED_VAR]]
  // CHECK: %[[SHOULD_CALL_VBASE_CTORS:.*]] = icmp ne i32 %[[IS_MOST_DERIVED_VAL]], 0
  // CHECK: br i1 %[[SHOULD_CALL_VBASE_CTORS]], label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]
  //
  // CHECK: [[INIT_VBASES]]
  // CHECK-NEXT: %[[vbptr_off:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 0
  // CHECK-NEXT: store ptr @"??_8D@constructors@@7B@", ptr %[[vbptr_off]]
  // CHECK-NEXT: getelementptr inbounds i8, ptr %{{.*}}, i32 4
  // CHECK-NEXT: call x86_thiscallcc ptr @"??0A@constructors@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
  // CHECK-NEXT: br label %[[SKIP_VBASES]]
  //
  // CHECK: [[SKIP_VBASES]]
  // CHECK: call x86_thiscallcc ptr @"??0C@constructors@@QAE@XZ"(ptr {{[^,]*}} %{{.*}}, i32 0)
  // CHECK: ret
}

struct E : virtual C {
  E();
};

E::E() {
  // CHECK: define dso_local x86_thiscallcc ptr @"??0E@constructors@@QAE@XZ"(ptr {{[^,]*}} returned {{[^,]*}} %this, i32 %is_most_derived) unnamed_addr
  // CHECK: store i32 %is_most_derived, ptr %[[IS_MOST_DERIVED_VAR:.*]], align 4
  // CHECK: %[[IS_MOST_DERIVED_VAL:.*]] = load i32, ptr %[[IS_MOST_DERIVED_VAR]]
  // CHECK: %[[SHOULD_CALL_VBASE_CTORS:.*]] = icmp ne i32 %[[IS_MOST_DERIVED_VAL]], 0
  // CHECK: br i1 %[[SHOULD_CALL_VBASE_CTORS]], label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]
  //
  // CHECK: [[INIT_VBASES]]
  // CHECK-NEXT: %[[offs:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 0
  // CHECK-NEXT: store ptr @"??_8E@constructors@@7B01@@", ptr %[[offs]]
  // CHECK-NEXT: %[[offs:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 4
  // CHECK-NEXT: store ptr @"??_8E@constructors@@7BC@1@@", ptr %[[offs]]
  // CHECK-NEXT: getelementptr inbounds i8, ptr %{{.*}}, i32 4
  // CHECK-NEXT: call x86_thiscallcc ptr @"??0A@constructors@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
  // CHECK: call x86_thiscallcc ptr @"??0C@constructors@@QAE@XZ"(ptr {{[^,]*}} %{{.*}}, i32 0)
  // CHECK-NEXT: br label %[[SKIP_VBASES]]
  //
  // CHECK: [[SKIP_VBASES]]
  // CHECK: ret
}

// PR16735 - even abstract classes should have a constructor emitted.
struct F {
  F();
  virtual void f() = 0;
};

F::F() {}
// CHECK: define dso_local x86_thiscallcc ptr @"??0F@constructors@@QAE@XZ"

} // end namespace constructors

namespace dtors {

struct A {
  ~A();
};

void call_nv_complete(A *a) {
  a->~A();
// CHECK: define dso_local void @"?call_nv_complete@dtors@@YAXPAUA@1@@Z"
// CHECK: call x86_thiscallcc void @"??1A@dtors@@QAE@XZ"
// CHECK: ret
}

// CHECK: declare dso_local x86_thiscallcc void @"??1A@dtors@@QAE@XZ"

// Now try some virtual bases, where we need the complete dtor.

struct B : virtual A { ~B(); };
struct C : virtual A { ~C(); };
struct D : B, C { ~D(); };

void call_vbase_complete(D *d) {
  d->~D();
// CHECK: define dso_local void @"?call_vbase_complete@dtors@@YAXPAUD@1@@Z"
// CHECK: call x86_thiscallcc void @"??_DD@dtors@@QAEXXZ"(ptr {{[^,]*}} %{{[^,]+}})
// CHECK: ret
}

// The complete dtor should call the base dtors for D and the vbase A (once).
// CHECK: define linkonce_odr dso_local x86_thiscallcc void @"??_DD@dtors@@QAEXXZ"({{.*}}) {{.*}} comdat
// CHECK-NOT: call
// CHECK: call x86_thiscallcc void @"??1D@dtors@@QAE@XZ"
// CHECK-NOT: call
// CHECK: call x86_thiscallcc void @"??1A@dtors@@QAE@XZ"
// CHECK-NOT: call
// CHECK: ret

void destroy_d_complete() {
  D d;
// CHECK: define dso_local void @"?destroy_d_complete@dtors@@YAXXZ"
// CHECK: call x86_thiscallcc void @"??_DD@dtors@@QAEXXZ"(ptr {{[^,]*}} %{{[^,]+}})
// CHECK: ret
}

// FIXME: Clang manually inlines the deletion, so we don't get a call to the
// deleting dtor (_G).  The only way to call deleting dtors currently is through
// a vftable.
void call_nv_deleting_dtor(D *d) {
  delete d;
// CHECK: define dso_local void @"?call_nv_deleting_dtor@dtors@@YAXPAUD@1@@Z"
// CHECK: call x86_thiscallcc void @"??_DD@dtors@@QAEXXZ"(ptr {{[^,]*}} %{{[^,]+}})
// CHECK: call void @"??3@YAXPAX@Z"
// CHECK: ret
}

}

namespace test1 {
struct A { };
struct B : virtual A {
  B(int *a);
  B(const char *a, ...);
  __cdecl B(short *a);
};
B::B(int *a) {}
B::B(const char *a, ...) {}
B::B(short *a) {}
// CHECK: define dso_local x86_thiscallcc ptr @"??0B@test1@@QAE@PAH@Z"
// CHECK:               (ptr {{[^,]*}} returned {{[^,]*}} %this, ptr %a, i32 %is_most_derived)
// CHECK: define dso_local ptr @"??0B@test1@@QAA@PBDZZ"
// CHECK:               (ptr {{[^,]*}} returned {{[^,]*}} %this, i32 %is_most_derived, ptr %a, ...)
// CHECK: define dso_local x86_thiscallcc ptr @"??0B@test1@@QAE@PAF@Z"
// CHECK:               (ptr {{[^,]*}} returned {{[^,]*}} %this, ptr %a, i32 %is_most_derived)

void construct_b() {
  int a;
  B b1(&a);
  B b2("%d %d", 1, 2);
}
// CHECK-LABEL: define dso_local void @"?construct_b@test1@@YAXXZ"()
// CHECK: call x86_thiscallcc ptr @"??0B@test1@@QAE@PAH@Z"
// CHECK:               (ptr {{.*}}, ptr {{.*}}, i32 1)
// CHECK: call ptr (ptr, i32, ptr, ...) @"??0B@test1@@QAA@PBDZZ"
// CHECK:               (ptr {{.*}}, i32 1, ptr {{.*}}, i32 1, i32 2)
}

namespace implicit_copy_vtable {
// This was a crash that only reproduced in ABIs without key functions.
struct ImplicitCopy {
  // implicit copy ctor
  virtual ~ImplicitCopy();
};
void CreateCopy(ImplicitCopy *a) {
  new ImplicitCopy(*a);
}
// CHECK: store {{.*}} @"??_7ImplicitCopy@implicit_copy_vtable@@6B@"

struct MoveOnly {
  MoveOnly(MoveOnly &&o) = default;
  virtual ~MoveOnly();
};
MoveOnly &&f();
void g() { new MoveOnly(f()); }
// CHECK: store {{.*}} @"??_7MoveOnly@implicit_copy_vtable@@6B@"
}

namespace delegating_ctor {
struct Y {};
struct X : virtual Y {
  X(int);
  X();
};
X::X(int) : X() {}
}
// CHECK: define dso_local x86_thiscallcc ptr @"??0X@delegating_ctor@@QAE@H@Z"(
// CHECK:  %[[is_most_derived_addr:.*]] = alloca i32, align 4
// CHECK:  store i32 %is_most_derived, ptr %[[is_most_derived_addr]]
// CHECK:  %[[is_most_derived:.*]] = load i32, ptr %[[is_most_derived_addr]]
// CHECK:  call x86_thiscallcc ptr @"??0X@delegating_ctor@@QAE@XZ"({{.*}} i32 %[[is_most_derived]])

// Dtor thunks for classes in anonymous namespaces should be internal, not
// linkonce_odr.
namespace {
struct A {
  virtual ~A() { }
};
}
void *getA() {
  return (void*)new A();
}
// CHECK: define internal x86_thiscallcc ptr @"??_GA@?A0x{{[^@]*}}@@UAEPAXI@Z"
// CHECK:               (ptr {{[^,]*}} %this, i32 %should_call_delete)
// CHECK: define internal x86_thiscallcc void @"??1A@?A0x{{[^@]*}}@@UAE@XZ"
// CHECK:               (ptr {{[^,]*}} %this)

// Check that we correctly transform __stdcall to __thiscall for ctors and
// dtors.
class G {
 public:
  __stdcall G() {};
// DTORS4: define linkonce_odr dso_local x86_thiscallcc ptr @"??0G@@QAE@XZ"
  __stdcall ~G() {};
// DTORS4: define linkonce_odr dso_local x86_thiscallcc void @"??1G@@QAE@XZ"
};

extern void testG() {
  G g;
}
