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
  // CHECK-LABEL: define dso_local x86_thiscallcc %struct.B* @"??0B@@QAE@XZ"
  // CHECK:   %[[THIS:.*]] = load %struct.B*, %struct.B**
  // CHECK:   br i1 %{{.*}}, label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]

  // Don't check the INIT_VBASES case as it's covered by the ctor tests.

  // CHECK: %[[SKIP_VBASES]]
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VBPTR:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 0
  // ...
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VFPTR_i8:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 %{{.*}}
  // CHECK:   %[[VFPTR:.*]] = bitcast i8* %[[VFPTR_i8]] to i32 (...)***
  // CHECK:   store i32 (...)** bitcast ({ [3 x i8*] }* @"??_7B@@6B@" to i32 (...)**), i32 (...)*** %[[VFPTR]]

  // Initialize vtorDisp:
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VBPTR:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 0
  // ...
  // CHECK:   %[[VBASE_OFFSET:.*]] = add nsw i32 0, %{{.*}}
  // CHECK:   %[[VTORDISP_VAL:.*]] = sub i32 %[[VBASE_OFFSET]], 8
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VBASE_i8:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 %[[VBASE_OFFSET]]
  // CHECK:   %[[VTORDISP_i8:.*]] = getelementptr i8, i8* %[[VBASE_i8]], i32 -4
  // CHECK:   %[[VTORDISP_PTR:.*]] = bitcast i8* %[[VTORDISP_i8]] to i32*
  // CHECK:   store i32 %[[VTORDISP_VAL]], i32* %[[VTORDISP_PTR]]

  // CHECK: ret
}

B::~B() {
  // CHECK-LABEL: define dso_local x86_thiscallcc void @"??1B@@UAE@XZ"
  // Store initial this:
  // CHECK:   %[[THIS_ADDR:.*]] = alloca %struct.B*
  // CHECK:   store %struct.B* %{{.*}}, %struct.B** %[[THIS_ADDR]], align 4
  // Reload and adjust the this parameter:
  // CHECK:   %[[THIS_RELOAD:.*]] = load %struct.B*, %struct.B** %[[THIS_ADDR]]
  // CHECK:   %[[THIS_UNADJ_i8:.*]] = bitcast %struct.B* %[[THIS_RELOAD]] to i8*
  // CHECK:   %[[THIS_ADJ_i8:.*]] = getelementptr inbounds i8, i8* %[[THIS_UNADJ_i8]], i32 -8
  // CHECK:   %[[THIS:.*]] = bitcast i8* %[[THIS_ADJ_i8]] to %struct.B*

  // Restore the vfptr that could have been changed by a subclass.
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VBPTR:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 0
  // ...
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VFPTR_i8:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 %{{.*}}
  // CHECK:   %[[VFPTR:.*]] = bitcast i8* %[[VFPTR_i8]] to i32 (...)***
  // CHECK:   store i32 (...)** bitcast ({ [3 x i8*] }* @"??_7B@@6B@" to i32 (...)**), i32 (...)*** %[[VFPTR]]

  // Initialize vtorDisp:
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VBPTR:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 0
  // ...
  // CHECK:   %[[VBASE_OFFSET:.*]] = add nsw i32 0, %{{.*}}
  // CHECK:   %[[VTORDISP_VAL:.*]] = sub i32 %[[VBASE_OFFSET]], 8
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VBASE_i8:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 %[[VBASE_OFFSET]]
  // CHECK:   %[[VTORDISP_i8:.*]] = getelementptr i8, i8* %[[VBASE_i8]], i32 -4
  // CHECK:   %[[VTORDISP_PTR:.*]] = bitcast i8* %[[VTORDISP_i8]] to i32*
  // CHECK:   store i32 %[[VTORDISP_VAL]], i32* %[[VTORDISP_PTR]]

  foo();  // Avoid the "trivial destructor" optimization.

  // CHECK: ret

  // CHECK2-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"??_DB@@QAEXXZ"(%struct.B*
  // CHECK2: %[[THIS:.*]] = load %struct.B*, %struct.B** {{.*}}
  // CHECK2: %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK2: %[[B_i8:.*]] = getelementptr i8, i8* %[[THIS_i8]], i32 8
  // CHECK2: %[[B:.*]] = bitcast i8* %[[B_i8]] to %struct.B*
  // CHECK2: call x86_thiscallcc void @"??1B@@UAE@XZ"(%struct.B* %[[B]])
  // CHECK2: %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK2: %[[VBASE_i8:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 8
  // CHECK2: %[[VBASE:.*]] = bitcast i8* %[[VBASE_i8]] to %struct.VBase*
  // CHECK2: call x86_thiscallcc void @"??1VBase@@UAE@XZ"(%struct.VBase* %[[VBASE]])
  // CHECK2: ret

  // CHECK2-LABEL: define linkonce_odr dso_local x86_thiscallcc i8* @"??_GB@@UAEPAXI@Z"
  // CHECK2:   store %struct.B* %{{.*}}, %struct.B** %[[THIS_ADDR:.*]], align 4
  // CHECK2:   %[[THIS:.*]] = load %struct.B*, %struct.B** %[[THIS_ADDR]]
  // CHECK2:   %[[THIS_PARAM_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK2:   %[[THIS_i8:.*]] = getelementptr inbounds i8, i8* %[[THIS_PARAM_i8:.*]], i32 -8
  // CHECK2:   %[[THIS:.*]] = bitcast i8* %[[THIS_i8]] to %struct.B*
  // CHECK2:   call x86_thiscallcc void @"??_DB@@QAEXXZ"(%struct.B* %[[THIS]])
  // ...
  // CHECK2: ret
}

void B::foo() {
// CHECK-LABEL: define dso_local x86_thiscallcc void @"?foo@B@@UAEXXZ"(i8*
//
// B::foo gets 'this' cast to VBase* in ECX (i.e. this+8) so we
// need to adjust 'this' before use.
//
// Coerce this to correct type:
// CHECK:   %[[THIS_STORE:.*]] = alloca %struct.B*
// CHECK:   %[[THIS_ADDR:.*]] = alloca %struct.B*
// CHECK:   %[[COERCE_VAL:.*]] = bitcast i8* %{{.*}} to %struct.B*
// CHECK:   store %struct.B* %[[COERCE_VAL]], %struct.B** %[[THIS_STORE]], align 4
//
// Store initial this:
// CHECK:   %[[THIS_INIT:.*]] = load %struct.B*, %struct.B** %[[THIS_STORE]]
// CHECK:   store %struct.B* %[[THIS_INIT]], %struct.B** %[[THIS_ADDR]], align 4
//
// Reload and adjust the this parameter:
// CHECK:   %[[THIS_RELOAD:.*]] = load %struct.B*, %struct.B** %[[THIS_ADDR]]
// CHECK:   %[[THIS_UNADJ_i8:.*]] = bitcast %struct.B* %[[THIS_RELOAD]] to i8*
// CHECK:   %[[THIS_ADJ_i8:.*]] = getelementptr inbounds i8, i8* %[[THIS_UNADJ_i8]], i32 -8
// CHECK:   %[[THIS:.*]] = bitcast i8* %[[THIS_ADJ_i8]] to %struct.B*

  field = 42;
// CHECK: %[[THIS8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8, i8* %[[THIS8]], i32 0
// CHECK: %[[VBPTR8:.*]] = bitcast i8* %[[VBPTR]] to i32**
// CHECK: %[[VBTABLE:.*]] = load i32*, i32** %[[VBPTR8]]
// CHECK: %[[VBENTRY:.*]] = getelementptr inbounds i32, i32* %[[VBTABLE]], i32 1
// CHECK: %[[VBOFFSET32:.*]] = load i32, i32* %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[THIS8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
// CHECK: %[[VBASE_i8:.*]] = getelementptr inbounds i8, i8* %[[THIS8]], i32 %[[VBOFFSET]]
// CHECK: %[[VBASE:.*]] = bitcast i8* %[[VBASE_i8]] to %struct.VBase*
// CHECK: %[[FIELD:.*]] = getelementptr inbounds %struct.VBase, %struct.VBase* %[[VBASE]], i32 0, i32 1
// CHECK: store i32 42, i32* %[[FIELD]], align 4
//
// CHECK: ret void
}

void call_vbase_bar(B *obj) {
// CHECK-LABEL: define dso_local void @"?call_vbase_bar@@YAXPAUB@@@Z"(%struct.B* %obj)
// CHECK: %[[OBJ:.*]] = load %struct.B

  obj->bar();
// When calling a vbase's virtual method, one needs to adjust 'this'
// at the caller site.
//
// CHECK: %[[OBJ_i8:.*]] = bitcast %struct.B* %[[OBJ]] to i8*
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8, i8* %[[OBJ_i8]], i32 0
// CHECK: %[[VBPTR8:.*]] = bitcast i8* %[[VBPTR]] to i32**
// CHECK: %[[VBTABLE:.*]] = load i32*, i32** %[[VBPTR8]]
// CHECK: %[[VBENTRY:.*]] = getelementptr inbounds i32, i32* %[[VBTABLE]], i32 1
// CHECK: %[[VBOFFSET32:.*]] = load i32, i32* %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[VBASE:.*]] = getelementptr inbounds i8, i8* %[[OBJ_i8]], i32 %[[VBOFFSET]]
//
// CHECK: %[[OBJ_i8:.*]] = bitcast %struct.B* %[[OBJ]] to i8*
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8, i8* %[[OBJ_i8]], i32 0
// CHECK: %[[VBPTR8:.*]] = bitcast i8* %[[VBPTR]] to i32**
// CHECK: %[[VBTABLE:.*]] = load i32*, i32** %[[VBPTR8]]
// CHECK: %[[VBENTRY:.*]] = getelementptr inbounds i32, i32* %[[VBTABLE]], i32 1
// CHECK: %[[VBOFFSET32:.*]] = load i32, i32* %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[VBASE_i8:.*]] = getelementptr inbounds i8, i8* %[[OBJ_i8]], i32 %[[VBOFFSET]]
// CHECK: %[[VFPTR:.*]] = bitcast i8* %[[VBASE_i8]] to void (i8*)***
// CHECK: %[[VFTABLE:.*]] = load void (i8*)**, void (i8*)*** %[[VFPTR]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds void (i8*)*, void (i8*)** %[[VFTABLE]], i64 2
// CHECK: %[[VFUN_VALUE:.*]] = load void (i8*)*, void (i8*)** %[[VFUN]]
//
// CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](i8* %[[VBASE]])
//
// CHECK: ret void
}

void delete_B(B *obj) {
// CHECK-LABEL: define dso_local void @"?delete_B@@YAXPAUB@@@Z"(%struct.B* %obj)
// CHECK: %[[OBJ:.*]] = load %struct.B

  delete obj;
// CHECK: %[[OBJ_i8:.*]] = bitcast %struct.B* %[[OBJ]] to i8*
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8, i8* %[[OBJ_i8]], i32 0
// CHECK: %[[VBPTR8:.*]] = bitcast i8* %[[VBPTR]] to i32**
// CHECK: %[[VBTABLE:.*]] = load i32*, i32** %[[VBPTR8]]
// CHECK: %[[VBENTRY:.*]] = getelementptr inbounds i32, i32* %[[VBTABLE]], i32 1
// CHECK: %[[VBOFFSET32:.*]] = load i32, i32* %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[VBASE_i8:.*]] = getelementptr inbounds i8, i8* %[[OBJ_i8]], i32 %[[VBOFFSET]]
// CHECK: %[[VBASE:.*]] = bitcast i8* %[[VBASE_i8]] to %struct.B*
//
// CHECK: %[[OBJ_i8:.*]] = bitcast %struct.B* %[[OBJ]] to i8*
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8, i8* %[[OBJ_i8]], i32 0
// CHECK: %[[VBPTR8:.*]] = bitcast i8* %[[VBPTR]] to i32**
// CHECK: %[[VBTABLE:.*]] = load i32*, i32** %[[VBPTR8]]
// CHECK: %[[VBENTRY:.*]] = getelementptr inbounds i32, i32* %[[VBTABLE]], i32 1
// CHECK: %[[VBOFFSET32:.*]] = load i32, i32* %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[VBASE_i8:.*]] = getelementptr inbounds i8, i8* %[[OBJ_i8]], i32 %[[VBOFFSET]]
// CHECK: %[[VFPTR:.*]] = bitcast i8* %[[VBASE_i8]] to i8* (%struct.B*, i32)***
// CHECK: %[[VFTABLE:.*]] = load i8* (%struct.B*, i32)**, i8* (%struct.B*, i32)*** %[[VFPTR]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds i8* (%struct.B*, i32)*, i8* (%struct.B*, i32)** %[[VFTABLE]], i64 0
// CHECK: %[[VFUN_VALUE:.*]] = load i8* (%struct.B*, i32)*, i8* (%struct.B*, i32)** %[[VFUN]]
//
// CHECK: call x86_thiscallcc i8* %[[VFUN_VALUE]](%struct.B* %[[VBASE]], i32 1)
// CHECK: ret void
}

void call_complete_dtor() {
  // CHECK-LABEL: define dso_local void @"?call_complete_dtor@@YAXXZ"
  B b;
  // CHECK: call x86_thiscallcc %struct.B* @"??0B@@QAE@XZ"(%struct.B* %[[B:.*]], i32 1)
  // CHECK-NOT: getelementptr
  // CHECK: call x86_thiscallcc void @"??_DB@@QAEXXZ"(%struct.B* %[[B]])
  // CHECK: ret
}

struct C : B {
  C();
  // has an implicit vdtor.
};

// Used to crash on an assertion.
C::C() {
// CHECK-LABEL: define dso_local x86_thiscallcc %struct.C* @"??0C@@QAE@XZ"
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
  // CHECK-LABEL: define dso_local x86_thiscallcc %"struct.multiple_vbases::D"* @"??0D@multiple_vbases@@QAE@XZ"
  // Just make sure we emit 3 vtordisps after initializing vfptrs.
  // CHECK: store i32 (...)** bitcast ({ [1 x i8*] }* @"??_7D@multiple_vbases@@6BA@1@@" to i32 (...)**), i32 (...)*** %{{.*}}
  // CHECK: store i32 (...)** bitcast ({ [1 x i8*] }* @"??_7D@multiple_vbases@@6BB@1@@" to i32 (...)**), i32 (...)*** %{{.*}}
  // CHECK: store i32 (...)** bitcast ({ [1 x i8*] }* @"??_7D@multiple_vbases@@6BC@1@@" to i32 (...)**), i32 (...)*** %{{.*}}
  // ...
  // CHECK: store i32 %{{.*}}, i32* %{{.*}}
  // CHECK: store i32 %{{.*}}, i32* %{{.*}}
  // CHECK: store i32 %{{.*}}, i32* %{{.*}}
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
  // CHECK-LABEL: define dso_local x86_thiscallcc void @"??1D@diamond@@UAE@XZ"(%"struct.diamond::D"*{{.*}})
  // Store initial this:
  // CHECK: %[[THIS_ADDR:.*]] = alloca %"struct.diamond::D"*
  // CHECK: store %"struct.diamond::D"* %{{.*}}, %"struct.diamond::D"** %[[THIS_ADDR]], align 4
  //
  // Reload and adjust the this parameter:
  // CHECK: %[[THIS_RELOAD:.*]] = load %"struct.diamond::D"*, %"struct.diamond::D"** %[[THIS_ADDR]]
  // CHECK: %[[THIS_UNADJ_i8:.*]] = bitcast %"struct.diamond::D"* %[[THIS_RELOAD]] to i8*
  // CHECK: %[[THIS_ADJ_i8:.*]] = getelementptr inbounds i8, i8* %[[THIS_UNADJ_i8]], i32 -24
  // CHECK: %[[THIS:.*]] = bitcast i8* %[[THIS_ADJ_i8]] to %"struct.diamond::D"*
  //
  // CHECK: %[[D_i8:.*]] = bitcast %"struct.diamond::D"* %[[THIS]] to i8*
  // CHECK: %[[C_i8:.*]] = getelementptr inbounds i8, i8* %[[D_i8]], i32 4
  // CHECK: %[[C:.*]] = bitcast i8* %[[C_i8]] to %"struct.diamond::C"*
  // CHECK: %[[C_i8:.*]] = bitcast %"struct.diamond::C"* %[[C]] to i8*
  // CHECK: %[[ARG_i8:.*]] = getelementptr i8, i8* %{{.*}}, i32 16
  // FIXME: We might consider changing the dtor this parameter type to i8*.
  // CHECK: %[[ARG:.*]] = bitcast i8* %[[ARG_i8]] to %"struct.diamond::C"*
  // CHECK: call x86_thiscallcc void @"??1C@diamond@@UAE@XZ"(%"struct.diamond::C"* %[[ARG]])

  // CHECK: %[[B:.*]] = bitcast %"struct.diamond::D"* %[[THIS]] to %"struct.diamond::B"*
  // CHECK: %[[B_i8:.*]] = bitcast %"struct.diamond::B"* %[[B]] to i8*
  // CHECK: %[[ARG_i8:.*]] = getelementptr i8, i8* %[[B_i8]], i32 4
  // CHECK: %[[ARG:.*]] = bitcast i8* %[[ARG_i8]] to %"struct.diamond::B"*
  // CHECK: call x86_thiscallcc void @"??1B@diamond@@UAE@XZ"(%"struct.diamond::B"* %[[ARG]])
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

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc %"struct.test2::C"* @"??0C@test2@@QAE@XZ"
// CHECK:           (%"struct.test2::C"* returned %this, i32 %is_most_derived)
// CHECK: br i1
//   Virtual bases
// CHECK: call x86_thiscallcc %"struct.test2::A"* @"??0A@test2@@QAE@XZ"(%"struct.test2::A"* %{{.*}})
// CHECK: br label
//   Non-virtual bases
// CHECK: call x86_thiscallcc %"struct.test2::B"* @"??0B@test2@@QAE@XZ"(%"struct.test2::B"* %{{.*}}, i32 0)
// CHECK: call x86_thiscallcc %"struct.test2::A"* @"??0A@test2@@QAE@XZ"(%"struct.test2::A"* %{{.*}})
// CHECK: ret

// CHECK2-LABEL: define linkonce_odr dso_local x86_thiscallcc %"struct.test2::B"* @"??0B@test2@@QAE@XZ"
// CHECK2:           (%"struct.test2::B"* returned %this, i32 %is_most_derived)
// CHECK2: call x86_thiscallcc %"struct.test2::A"* @"??0A@test2@@QAE@XZ"(%"struct.test2::A"* %{{.*}})
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
  // CHECK-LABEL: define dso_local x86_thiscallcc void @"?bar@D@test3@@UAEXXZ"(%"struct.test3::D"* %this)

  C::foo();
  // Shouldn't need any vbtable lookups.  All we have to do is adjust to C*,
  // then compensate for the adjustment performed in the C::foo() prologue.
  // CHECK-NOT: load i8*, i8**
  // CHECK: %[[OBJ_i8:.*]] = bitcast %"struct.test3::D"* %{{.*}} to i8*
  // CHECK: %[[C_i8:.*]] = getelementptr inbounds i8, i8* %[[OBJ_i8]], i32 8
  // CHECK: %[[C:.*]] = bitcast i8* %[[C_i8]] to %"struct.test3::C"*
  // CHECK: %[[C_i8:.*]] = bitcast %"struct.test3::C"* %[[C]] to i8*
  // CHECK: %[[ARG:.*]] = getelementptr i8, i8* %[[C_i8]], i32 4
  // CHECK: call x86_thiscallcc void @"?foo@C@test3@@UAEXXZ"(i8* %[[ARG]])
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
  // CHECK-LABEL: define dso_local x86_thiscallcc void @"??1C@test4@@UAE@XZ"(%"struct.test4::C"* %this)

  // In this case "this" points to the most derived class, so no GEPs needed.
  // CHECK-NOT: getelementptr
  // CHECK-NOT: bitcast
  // CHECK: %[[VFPTR_i8:.*]] = bitcast %"struct.test4::C"* %{{.*}} to i32 (...)***
  // CHECK: store i32 (...)** bitcast ({ [1 x i8*] }* @"??_7C@test4@@6BB@1@@" to i32 (...)**), i32 (...)*** %[[VFPTR_i8]]

  foo(this);
  // CHECK: ret
}

void destroy(C *obj) {
  // CHECK-LABEL: define dso_local void @"?destroy@test4@@YAXPAUC@1@@Z"(%"struct.test4::C"* %obj)

  delete obj;
  // CHECK: %[[VPTR:.*]] = bitcast %"struct.test4::C"* %[[OBJ:.*]] to i8* (%"struct.test4::C"*, i32)***
  // CHECK: %[[VFTABLE:.*]] = load i8* (%"struct.test4::C"*, i32)**, i8* (%"struct.test4::C"*, i32)*** %[[VPTR]]
  // CHECK: %[[VFTENTRY:.*]] = getelementptr inbounds i8* (%"struct.test4::C"*, i32)*, i8* (%"struct.test4::C"*, i32)** %[[VFTABLE]], i64 0
  // CHECK: %[[VFUN:.*]] = load i8* (%"struct.test4::C"*, i32)*, i8* (%"struct.test4::C"*, i32)** %[[VFTENTRY]]
  // CHECK: call x86_thiscallcc i8* %[[VFUN]](%"struct.test4::C"* %[[OBJ]], i32 1)
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
  // CHECK-LABEL: define dso_local x86_thiscallcc void @"??1E@test4@@UAE@XZ"(%"struct.test4::E"* %this)

  // In this case "this" points to the most derived class, so no GEPs needed.
  // CHECK-NOT: getelementptr
  // CHECK-NOT: bitcast
  // CHECK: %[[VFPTR_i8:.*]] = bitcast %"struct.test4::E"* %{{.*}} to i32 (...)***
  // CHECK: store i32 (...)** bitcast ({ [1 x i8*] }* @"??_7E@test4@@6BD@1@@" to i32 (...)**), i32 (...)*** %[[VFPTR_i8]]
  foo(this);
}

void destroy(E *obj) {
  // CHECK-LABEL: define dso_local void @"?destroy@test4@@YAXPAUE@1@@Z"(%"struct.test4::E"* %obj)

  // CHECK-NOT: getelementptr
  // CHECK: %[[OBJ_i8:.*]] = bitcast %"struct.test4::E"* %[[OBJ]] to i8*
  // CHECK: %[[B_i8:.*]] = getelementptr inbounds i8, i8* %[[OBJ_i8]], i32 4
  // FIXME: in fact, the call should take i8* and the bitcast is redundant.
  // CHECK: %[[B_as_E:.*]] = bitcast i8* %[[B_i8]] to %"struct.test4::E"*
  // CHECK: %[[OBJ_i8:.*]] = bitcast %"struct.test4::E"* %[[OBJ:.*]] to i8*
  // CHECK: %[[B_i8:.*]] = getelementptr inbounds i8, i8* %[[OBJ_i8]], i32 4
  // CHECK: %[[VPTR:.*]] = bitcast i8* %[[B_i8]] to i8* (%"struct.test4::E"*, i32)***
  // CHECK: %[[VFTABLE:.*]] = load i8* (%"struct.test4::E"*, i32)**, i8* (%"struct.test4::E"*, i32)*** %[[VPTR]]
  // CHECK: %[[VFTENTRY:.*]] = getelementptr inbounds i8* (%"struct.test4::E"*, i32)*, i8* (%"struct.test4::E"*, i32)** %[[VFTABLE]], i64 0
  // CHECK: %[[VFUN:.*]] = load i8* (%"struct.test4::E"*, i32)*, i8* (%"struct.test4::E"*, i32)** %[[VFTENTRY]]
  // CHECK: call x86_thiscallcc i8* %[[VFUN]](%"struct.test4::E"* %[[B_as_E]], i32 1)
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
// CHECK-LABEL: define dso_local x86_thiscallcc %"struct.test5::C"* @"??0C@test5@@QAE@XZ"(
// CHECK:   %[[THIS:.*]] = load %"struct.test5::C"*, %"struct.test5::C"**
// CHECK:   br i1 %{{.*}}, label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]

// CHECK: %[[SKIP_VBASES]]
// CHECK:   %[[B:.*]] = bitcast %"struct.test5::C"* %[[THIS]] to %"struct.test5::B"*
// CHECK:   %[[B_i8:.*]] = bitcast %"struct.test5::B"* %[[B]] to i8*
// CHECK:   %[[FIELD:.*]] = getelementptr inbounds i8, i8* %[[B_i8]], i32 4
// CHECK:   call void @llvm.memset.p0i8.i32(i8* align 4 %[[FIELD]], i8 0, i32 4, i1 false)
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
// CHECK-LABEL: define dso_local void @"?callit@pr27621@@YAXPAUC@1@@Z"(%"struct.pr27621::C"* %{{.*}})
// CHECK: %[[B_i8:.*]] = getelementptr i8, i8* %{{.*}}, i32 4
// CHECK: call x86_thiscallcc void @"?g@C@pr27621@@UAEXXZ"(i8* %[[B_i8]])
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
// CHECK-LABEL: define dso_local x86_thiscallcc %"class.test6::D"* @"??0D@test6@@AAE@XZ"(
// CHECK:   %[[THIS:.*]] = load %"class.test6::D"*, %"class.test6::D"**
// CHECK:   br i1 %{{.*}}, label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]

// CHECK: %[[SKIP_VBASES]]
// CHECK:   %[[C:.*]] = bitcast %"class.test6::D"* %[[THIS]] to %"class.test6::C"*
// CHECK:   %[[C_i8:.*]] = bitcast %"class.test6::C"* %[[C]] to i8*
// CHECK:   %[[FIELD:.*]] = getelementptr inbounds i8, i8* %[[C_i8]], i32 8
// CHECK:   call void @llvm.memset.p0i8.i32(i8* align 4 %[[FIELD]], i8 0, i32 4, i1 false)
}
