// RUN: %clang_cc1 %s -fno-rtti -std=c++11 -Wno-inaccessible-base -triple=i386-pc-win32 -emit-llvm -o %t
// RUN: FileCheck %s < %t
// RUN: FileCheck --check-prefix=CHECK2 %s < %t

// For now, just make sure x86_64 doesn't crash.
// RUN: %clang_cc1 %s -fno-rtti -std=c++11 -Wno-inaccessible-base -triple=x86_64-pc-win32 -emit-llvm -o %t

struct VBase {
  virtual ~VBase();
  virtual void foo();
  virtual void bar();
  int field;
};

struct B : virtual VBase {
  B();
  virtual ~B();
  virtual void foo();
  virtual void bar();
};

B::B() {
  // CHECK-LABEL: define dso_local x86_thiscallcc noundef ptr @"??0B@@QAE@XZ"
  // CHECK:   %[[THIS:.*]] = load ptr, ptr
  // CHECK:   br i1 %{{.*}}, label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]

  // Don't check the INIT_VBASES case as it's covered by the ctor tests.

  // CHECK: %[[SKIP_VBASES]]
  // CHECK:   %[[VBPTR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i32 0
  // ...
  // CHECK:   %[[VFPTR_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i32 %{{.*}}
  // CHECK:   store ptr @"??_7B@@6B@", ptr %[[VFPTR_i8]]

  // Initialize vtorDisp:
  // CHECK:   %[[VBPTR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i32 0
  // ...
  // CHECK:   %[[VBASE_OFFSET:.*]] = add nsw i32 0, %{{.*}}
  // CHECK:   %[[VTORDISP_VAL:.*]] = sub i32 %[[VBASE_OFFSET]], 8
  // CHECK:   %[[VBASE_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i32 %[[VBASE_OFFSET]]
  // CHECK:   %[[VTORDISP_i8:.*]] = getelementptr i8, ptr %[[VBASE_i8]], i32 -4
  // CHECK:   store i32 %[[VTORDISP_VAL]], ptr %[[VTORDISP_i8]]

  // CHECK: ret
}

B::~B() {
  // CHECK-LABEL: define dso_local x86_thiscallcc void @"??1B@@UAE@XZ"
  // Store initial this:
  // CHECK:   %[[THIS_ADDR:.*]] = alloca ptr
  // CHECK:   store ptr %{{.*}}, ptr %[[THIS_ADDR]], align 4
  // Reload and adjust the this parameter:
  // CHECK:   %[[THIS_RELOAD:.*]] = load ptr, ptr %[[THIS_ADDR]]
  // CHECK:   %[[THIS_ADJ_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS_RELOAD]], i32 -8

  // Restore the vfptr that could have been changed by a subclass.
  // CHECK:   %[[VBPTR:.*]] = getelementptr inbounds i8, ptr %[[THIS_ADJ_i8]], i32 0
  // ...
  // CHECK:   %[[VFPTR_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS_ADJ_i8]], i32 %{{.*}}
  // CHECK:   store ptr @"??_7B@@6B@", ptr %[[VFPTR_i8]]

  // Initialize vtorDisp:
  // CHECK:   %[[VBPTR:.*]] = getelementptr inbounds i8, ptr %[[THIS_ADJ_i8]], i32 0
  // ...
  // CHECK:   %[[VBASE_OFFSET:.*]] = add nsw i32 0, %{{.*}}
  // CHECK:   %[[VTORDISP_VAL:.*]] = sub i32 %[[VBASE_OFFSET]], 8
  // CHECK:   %[[VBASE_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS_ADJ_i8]], i32 %[[VBASE_OFFSET]]
  // CHECK:   %[[VTORDISP_i8:.*]] = getelementptr i8, ptr %[[VBASE_i8]], i32 -4
  // CHECK:   store i32 %[[VTORDISP_VAL]], ptr %[[VTORDISP_i8]]

  foo();  // Avoid the "trivial destructor" optimization.

  // CHECK: ret

  // CHECK2-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"??_DB@@QAEXXZ"(ptr
  // CHECK2: %[[THIS:.*]] = load ptr, ptr {{.*}}
  // CHECK2: %[[B_i8:.*]] = getelementptr i8, ptr %[[THIS]], i32 8
  // CHECK2: call x86_thiscallcc void @"??1B@@UAE@XZ"(ptr{{[^,]*}} %[[B_i8]])
  // CHECK2: %[[VBASE_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i32 8
  // CHECK2: call x86_thiscallcc void @"??1VBase@@UAE@XZ"(ptr {{[^,]*}} %[[VBASE_i8]])
  // CHECK2: ret

  // CHECK2-LABEL: define linkonce_odr dso_local x86_thiscallcc noundef ptr @"??_GB@@UAEPAXI@Z"
  // CHECK2:   store ptr %{{.*}}, ptr %[[THIS_ADDR:.*]], align 4
  // CHECK2:   %[[THIS_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS_PARAM_i8:.*]], i32 -8
  // CHECK2:   call x86_thiscallcc void @"??_DB@@QAEXXZ"(ptr {{[^,]*}} %[[THIS_i8]])
  // ...
  // CHECK2: ret
}

void B::foo() {
// CHECK-LABEL: define dso_local x86_thiscallcc void @"?foo@B@@UAEXXZ"(ptr
//
// B::foo gets 'this' cast to VBase* in ECX (i.e. this+8) so we
// need to adjust 'this' before use.
//
// Coerce this to correct type:
// CHECK:   %[[THIS_ADDR:.*]] = alloca ptr
//
// Store initial this:
// CHECK:   store ptr {{.*}}, ptr %[[THIS_ADDR]], align 4
//
// Reload and adjust the this parameter:
// CHECK:   %[[THIS_RELOAD:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK:   %[[THIS_ADJ_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS_RELOAD]], i32 -8

  field = 42;
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8, ptr %[[THIS_ADJ_i8]], i32 0
// CHECK: %[[VBTABLE:.*]] = load ptr, ptr %[[VBPTR]]
// CHECK: %[[VBENTRY:.*]] = getelementptr inbounds i32, ptr %[[VBTABLE]], i32 1
// CHECK: %[[VBOFFSET32:.*]] = load i32, ptr %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[VBASE_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS_ADJ_i8]], i32 %[[VBOFFSET]]
// CHECK: %[[FIELD:.*]] = getelementptr inbounds %struct.VBase, ptr %[[VBASE_i8]], i32 0, i32 1
// CHECK: store i32 42, ptr %[[FIELD]], align 4
//
// CHECK: ret void
}

void call_vbase_bar(B *obj) {
// CHECK-LABEL: define dso_local void @"?call_vbase_bar@@YAXPAUB@@@Z"(ptr noundef %obj)
// CHECK: %[[OBJ:.*]] = load ptr

  obj->bar();
// When calling a vbase's virtual method, one needs to adjust 'this'
// at the caller site.
//
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8, ptr %[[OBJ]], i32 0
// CHECK: %[[VBTABLE:.*]] = load ptr, ptr %[[VBPTR]]
// CHECK: %[[VBENTRY:.*]] = getelementptr inbounds i32, ptr %[[VBTABLE]], i32 1
// CHECK: %[[VBOFFSET32:.*]] = load i32, ptr %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[VBASE:.*]] = getelementptr inbounds i8, ptr %[[OBJ]], i32 %[[VBOFFSET]]
//
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8, ptr %[[OBJ]], i32 0
// CHECK: %[[VBTABLE:.*]] = load ptr, ptr %[[VBPTR]]
// CHECK: %[[VBENTRY:.*]] = getelementptr inbounds i32, ptr %[[VBTABLE]], i32 1
// CHECK: %[[VBOFFSET32:.*]] = load i32, ptr %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[VBASE_i8:.*]] = getelementptr inbounds i8, ptr %[[OBJ]], i32 %[[VBOFFSET]]
// CHECK: %[[VFTABLE:.*]] = load ptr, ptr %[[VBASE_i8]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds ptr, ptr %[[VFTABLE]], i64 2
// CHECK: %[[VFUN_VALUE:.*]] = load ptr, ptr %[[VFUN]]
//
// CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](ptr noundef %[[VBASE]])
//
// CHECK: ret void
}

void delete_B(B *obj) {
// CHECK-LABEL: define dso_local void @"?delete_B@@YAXPAUB@@@Z"(ptr noundef %obj)
// CHECK: %[[OBJ:.*]] = load ptr

  delete obj;
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8, ptr %[[OBJ]], i32 0
// CHECK: %[[VBTABLE:.*]] = load ptr, ptr %[[VBPTR]]
// CHECK: %[[VBENTRY:.*]] = getelementptr inbounds i32, ptr %[[VBTABLE]], i32 1
// CHECK: %[[VBOFFSET32:.*]] = load i32, ptr %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[VBASE:.*]] = getelementptr inbounds i8, ptr %[[OBJ]], i32 %[[VBOFFSET]]
//
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8, ptr %[[OBJ]], i32 0
// CHECK: %[[VBTABLE:.*]] = load ptr, ptr %[[VBPTR]]
// CHECK: %[[VBENTRY:.*]] = getelementptr inbounds i32, ptr %[[VBTABLE]], i32 1
// CHECK: %[[VBOFFSET32:.*]] = load i32, ptr %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[VBASE_i8:.*]] = getelementptr inbounds i8, ptr %[[OBJ]], i32 %[[VBOFFSET]]
// CHECK: %[[VFTABLE:.*]] = load ptr, ptr %[[VBASE_i8]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds ptr, ptr %[[VFTABLE]], i64 0
// CHECK: %[[VFUN_VALUE:.*]] = load ptr, ptr %[[VFUN]]
//
// CHECK: call x86_thiscallcc noundef ptr %[[VFUN_VALUE]](ptr {{[^,]*}} %[[VBASE]], i32 noundef 1)
// CHECK: ret void
}

void call_complete_dtor() {
  // CHECK-LABEL: define dso_local void @"?call_complete_dtor@@YAXXZ"
  B b;
  // CHECK: call x86_thiscallcc noundef ptr @"??0B@@QAE@XZ"(ptr {{[^,]*}} %[[B:.*]], i32 noundef 1)
  // CHECK-NOT: getelementptr
  // CHECK: call x86_thiscallcc void @"??_DB@@QAEXXZ"(ptr {{[^,]*}} %[[B]])
  // CHECK: ret
}

struct C : B {
  C();
  // has an implicit vdtor.
};

// Used to crash on an assertion.
C::C() {
// CHECK-LABEL: define dso_local x86_thiscallcc noundef ptr @"??0C@@QAE@XZ"
}

namespace multiple_vbases {
struct A {
  virtual void a();
};

struct B {
  virtual void b();
};

struct C {
  virtual void c();
};

struct D : virtual A, virtual B, virtual C {
  virtual void a();
  virtual void b();
  virtual void c();
  D();
};

D::D() {
  // CHECK-LABEL: define dso_local x86_thiscallcc noundef ptr @"??0D@multiple_vbases@@QAE@XZ"
  // Just make sure we emit 3 vtordisps after initializing vfptrs.
  // CHECK: store ptr @"??_7D@multiple_vbases@@6BA@1@@", ptr %{{.*}}
  // CHECK: store ptr @"??_7D@multiple_vbases@@6BB@1@@", ptr %{{.*}}
  // CHECK: store ptr @"??_7D@multiple_vbases@@6BC@1@@", ptr %{{.*}}
  // ...
  // CHECK: store i32 %{{.*}}, ptr %{{.*}}
  // CHECK: store i32 %{{.*}}, ptr %{{.*}}
  // CHECK: store i32 %{{.*}}, ptr %{{.*}}
  // CHECK: ret
}
}

namespace diamond {
struct A {
  A();
  virtual ~A();
};

struct B : virtual A {
  B();
  ~B();
};

struct C : virtual A {
  C();
  ~C();
  int c1, c2, c3;
};

struct Z {
  int z;
};

struct D : virtual Z, B, C {
  D();
  ~D();
} d;

D::~D() {
  // CHECK-LABEL: define dso_local x86_thiscallcc void @"??1D@diamond@@UAE@XZ"(ptr{{.*}})
  // Store initial this:
  // CHECK: %[[THIS_ADDR:.*]] = alloca ptr
  // CHECK: store ptr %{{.*}}, ptr %[[THIS_ADDR]], align 4
  //
  // Reload and adjust the this parameter:
  // CHECK: %[[THIS_RELOAD:.*]] = load ptr, ptr %[[THIS_ADDR]]
  // CHECK: %[[THIS_ADJ_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS_RELOAD]], i32 -24
  //
  // CHECK: %[[C_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS_ADJ_i8]], i32 4
  // CHECK: %[[ARG:.*]] = getelementptr i8, ptr %{{.*}}, i32 16
  // CHECK: call x86_thiscallcc void @"??1C@diamond@@UAE@XZ"(ptr{{[^,]*}} %[[ARG]])

  // CHECK: %[[ARG:.*]] = getelementptr i8, ptr %[[THIS_ADJ_i8]], i32 4
  // CHECK: call x86_thiscallcc void @"??1B@diamond@@UAE@XZ"(ptr{{[^,]*}} %[[ARG]])
  // CHECK: ret void
}

}

namespace test2 {
struct A { A(); };
struct B : virtual A { B() {} };
struct C : B, A { C() {} };

// PR18435: Order mattered here.  We were generating code for the delegating
// call to B() from C().
void callC() { C x; }

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc noundef ptr @"??0C@test2@@QAE@XZ"
// CHECK:           (ptr {{[^,]*}} returned align 4 dereferenceable(8) %this, i32 noundef %is_most_derived)
// CHECK: br i1
//   Virtual bases
// CHECK: call x86_thiscallcc noundef ptr @"??0A@test2@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
// CHECK: br label
//   Non-virtual bases
// CHECK: call x86_thiscallcc noundef ptr @"??0B@test2@@QAE@XZ"(ptr {{[^,]*}} %{{.*}}, i32 noundef 0)
// CHECK: call x86_thiscallcc noundef ptr @"??0A@test2@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
// CHECK: ret

// CHECK2-LABEL: define linkonce_odr dso_local x86_thiscallcc noundef ptr @"??0B@test2@@QAE@XZ"
// CHECK2:           (ptr {{[^,]*}} returned align 4 dereferenceable(4) %this, i32 noundef %is_most_derived)
// CHECK2: call x86_thiscallcc noundef ptr @"??0A@test2@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
// CHECK2: ret

}

namespace test3 {
// PR19104: A non-virtual call of a virtual method doesn't use vftable thunks,
// so requires only static adjustment which is different to the one used
// for virtual calls.
struct A {
  virtual void foo();
};

struct B : virtual A {
  virtual void bar();
};

struct C : virtual A {
  virtual void foo();
};

struct D : B, C {
  virtual void bar();
  int field;  // Laid out between C and A subobjects in D.
};

void D::bar() {
  // CHECK-LABEL: define dso_local x86_thiscallcc void @"?bar@D@test3@@UAEXXZ"(ptr {{[^,]*}} %this)

  C::foo();
  // Shouldn't need any vbtable lookups.  All we have to do is adjust to C*,
  // then compensate for the adjustment performed in the C::foo() prologue.
  // CHECK: %[[C_i8:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 8
  // CHECK: %[[ARG:.*]] = getelementptr i8, ptr %[[C_i8]], i32 4
  // CHECK: call x86_thiscallcc void @"?foo@C@test3@@UAEXXZ"(ptr noundef %[[ARG]])
  // CHECK: ret
}
}

namespace test4{
// PR19172: We used to merge method vftable locations wrong.

struct A {
  virtual ~A() {}
};

struct B {
  virtual ~B() {}
};

struct C : virtual A, B {
  virtual ~C();
};

void foo(void*);

C::~C() {
  // CHECK-LABEL: define dso_local x86_thiscallcc void @"??1C@test4@@UAE@XZ"(ptr {{[^,]*}} %this)

  // In this case "this" points to the most derived class, so no GEPs needed.
  // CHECK-NOT: getelementptr
  // CHECK: store ptr @"??_7C@test4@@6BB@1@@", ptr %{{.*}}

  foo(this);
  // CHECK: ret
}

void destroy(C *obj) {
  // CHECK-LABEL: define dso_local void @"?destroy@test4@@YAXPAUC@1@@Z"(ptr noundef %obj)

  delete obj;
  // CHECK: %[[OBJ:.*]] = load ptr, ptr
  // CHECK: %[[VFTABLE:.*]] = load ptr, ptr %[[OBJ]]
  // CHECK: %[[VFTENTRY:.*]] = getelementptr inbounds ptr, ptr %[[VFTABLE]], i64 0
  // CHECK: %[[VFUN:.*]] = load ptr, ptr %[[VFTENTRY]]
  // CHECK: call x86_thiscallcc noundef ptr %[[VFUN]](ptr {{[^,]*}} %[[OBJ]], i32 noundef 1)
  // CHECK: ret
}

struct D {
  virtual void d();
};

// The first non-virtual base doesn't have a vdtor,
// but "this adjustment" is not needed.
struct E : D, B, virtual A {
  virtual ~E();
};

E::~E() {
  // CHECK-LABEL: define dso_local x86_thiscallcc void @"??1E@test4@@UAE@XZ"(ptr{{[^,]*}} %this)

  // In this case "this" points to the most derived class, so no GEPs needed.
  // CHECK-NOT: getelementptr
  // CHECK: store ptr @"??_7E@test4@@6BD@1@@", ptr %{{.*}}
  foo(this);
}

void destroy(E *obj) {
  // CHECK-LABEL: define dso_local void @"?destroy@test4@@YAXPAUE@1@@Z"(ptr noundef %obj)

  // CHECK-NOT: getelementptr
  // CHECK: %[[THIS_i8:.*]] = getelementptr inbounds i8, ptr %[[OBJ]], i32 4
  // CHECK: %[[B_i8:.*]] = getelementptr inbounds i8, ptr %[[OBJ:.*]], i32 4
  // CHECK: %[[VFTABLE:.*]] = load ptr, ptr %[[B_i8]]
  // CHECK: %[[VFTENTRY:.*]] = getelementptr inbounds ptr, ptr %[[VFTABLE]], i64 0
  // CHECK: %[[VFUN:.*]] = load ptr, ptr %[[VFTENTRY]]
  // CHECK: call x86_thiscallcc noundef ptr %[[VFUN]](ptr{{[^,]*}} %[[THIS_i8]], i32 noundef 1)
  delete obj;
}

}

namespace test5 {
// PR25370: Don't zero-initialize vbptrs in virtual bases.
struct A {
  virtual void f();
};

struct B : virtual A {
  int Field;
};

struct C : B {
  C();
};

C::C() : B() {}
// CHECK-LABEL: define dso_local x86_thiscallcc noundef ptr @"??0C@test5@@QAE@XZ"(
// CHECK:   %[[THIS:.*]] = load ptr, ptr
// CHECK:   br i1 %{{.*}}, label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]

// CHECK: %[[SKIP_VBASES]]
// CHECK:   %[[FIELD:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i32 4
// CHECK:   call void @llvm.memset.p0.i32(ptr align 4 %[[FIELD]], i8 0, i32 4, i1 false)
}

namespace pr27621 {
// Devirtualization through a static_cast used to make us compute the 'this'
// adjustment for B::g instead of C::g. When we directly call C::g, 'this' is a
// B*, and the prologue of C::g will adjust it to a C*.
struct A { virtual void f(); };
struct B { virtual void g(); };
struct C final : A, B {
  virtual void h();
  void g() override;
};
void callit(C *p) {
  static_cast<B*>(p)->g();
}
// CHECK-LABEL: define dso_local void @"?callit@pr27621@@YAXPAUC@1@@Z"(ptr noundef %{{.*}})
// CHECK: %[[B_i8:.*]] = getelementptr i8, ptr %{{.*}}, i32 4
// CHECK: call x86_thiscallcc void @"?g@C@pr27621@@UAEXXZ"(ptr noundef %[[B_i8]])
}

namespace test6 {
class A {};
class B : virtual A {};
class C : virtual B {
  virtual void m_fn1();
  float field;
};
class D : C {
  D();
};
D::D() : C() {}
// CHECK-LABEL: define dso_local x86_thiscallcc noundef ptr @"??0D@test6@@AAE@XZ"(
// CHECK:   %[[THIS:.*]] = load ptr, ptr
// CHECK:   br i1 %{{.*}}, label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]

// CHECK: %[[SKIP_VBASES]]
// CHECK:   %[[FIELD:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i32 8
// CHECK:   call void @llvm.memset.p0.i32(ptr align 4 %[[FIELD]], i8 0, i32 4, i1 false)
}

namespace pr36921 {
struct A {
  virtual ~A() {}
};
struct B {
  virtual ~B() {}
};
struct C : virtual B {};
struct D : virtual A, C {};
D d;
// CHECK2-LABEL: define linkonce_odr dso_local x86_thiscallcc noundef ptr @"??_GD@pr36921@@UAEPAXI@Z"(
// CHECK2:   %[[THIS_RELOAD:.*]] = load ptr, ptr
// CHECK2:   %[[THIS_ADJ_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS_RELOAD]], i32 -4
}

namespace issue_60465 {
// We used to assume the first argument to all destructors was the derived type
// even when there was a 'this' adjustment.
struct A {
  virtual ~A();
};

struct alignas(2 * sizeof(void *)) B : virtual A {
  ~B();
  void *x, *y;
};

B::~B() {
// The 'this' parameter should not have a type of ptr and
// must not have 'align 8', since at least B's copy of A is only 'align 4'.
// CHECK-LABEL: define dso_local x86_thiscallcc void @"??1B@issue_60465@@UAE@XZ"(ptr noundef %this)
// CHECK:   %[[THIS_ADJ_i8:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 -12
// CHECK:   %[[X:.*]] = getelementptr inbounds %"struct.issue_60465::B", ptr %[[THIS_ADJ_i8]], i32 0, i32 1
// CHECK:   store ptr null, ptr %[[X]], align 4
// CHECK:   %[[Y:.*]] = getelementptr inbounds %"struct.issue_60465::B", ptr %[[THIS_ADJ_i8]], i32 0, i32 2
// CHECK:   store ptr null, ptr %[[Y]], align 8
  x = nullptr;
  y = nullptr;
}
}
