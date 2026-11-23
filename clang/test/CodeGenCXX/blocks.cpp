// RUN: %clang_cc1 %s -fblocks -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s

// CHECK: @[[BLOCK_DESCRIPTOR22:.*]] = internal constant { i64, i64, ptr, ptr, ptr, ptr } { i64 0, i64 36, ptr @__copy_helper_block_8_32c22_ZTSN12_GLOBAL__N_11BE, ptr @__destroy_helper_block_8_32c22_ZTSN12_GLOBAL__N_11BE, ptr @{{.*}}, ptr null }, align 8

namespace test0 {
  // CHECK-LABEL: define{{.*}} void @_ZN5test04testEi(
  // CHECK: define internal void @___ZN5test04testEi_block_invoke{{.*}}(
  // CHECK: define internal void @___ZN5test04testEi_block_invoke_2{{.*}}(
  void test(int x) {
    ^{ ^{ (void) x; }; };
  }
}

extern void (^out)();

namespace test1 {
  // Capturing const objects doesn't require a local block.
  // CHECK-LABEL: define{{.*}} void @_ZN5test15test1Ev()
  // CHECK:   store ptr @__block_literal_global{{.*}}, ptr @out
  void test1() {
    const int NumHorsemen = 4;
    out = ^{ (void) NumHorsemen; };
  }

  // That applies to structs too...
  // CHECK-LABEL: define{{.*}} void @_ZN5test15test2Ev()
  // CHECK:   store ptr @__block_literal_global{{.*}}, ptr @out
  struct loc { double x, y; };
  void test2() {
    const loc target = { 5, 6 };
    out = ^{ (void) target; };
  }

  // ...unless they have mutable fields...
  // CHECK-LABEL: define{{.*}} void @_ZN5test15test3Ev()
  // CHECK:   [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK:   store ptr [[BLOCK]], ptr @out
  struct mut { mutable int x; };
  void test3() {
    const mut obj = { 5 };
    out = ^{ (void) obj; };
  }

  // ...or non-trivial destructors...
  // CHECK-LABEL: define{{.*}} void @_ZN5test15test4Ev()
  // CHECK:   [[OBJ:%.*]] = alloca
  // CHECK:   [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK:   store ptr [[BLOCK]], ptr @out
  struct scope { int x; ~scope(); };
  void test4() {
    const scope obj = { 5 };
    out = ^{ (void) obj; };
  }

  // ...or non-trivial copy constructors, but it's not clear how to do
  // that and still have a constant initializer in '03.
}

namespace test2 {
  struct A {
    A();
    A(const A &);
    ~A();
  };

  struct B {
    B();
    B(const B &);
    ~B();
  };

  // CHECK-LABEL: define{{.*}} void @_ZN5test24testEv()
  void test() {
    __block A a;
    __block B b;
    ^{ (void)a; (void)b; };
  }

  // CHECK-LABEL: define internal void @__Block_byref_object_copy
  // CHECK: call void @_ZN5test21AC1ERKS0_(

  // CHECK-LABEL: define internal void @__Block_byref_object_dispose
  // CHECK: call void @_ZN5test21AD1Ev(

  // CHECK-LABEL: define internal void @__Block_byref_object_copy
  // CHECK: call void @_ZN5test21BC1ERKS0_(

  // CHECK-LABEL: define internal void @__Block_byref_object_dispose
  // CHECK: call void @_ZN5test21BD1Ev(
}

// Make sure we mark destructors for parameters captured in blocks.
namespace test3 {
  struct A {
    A(const A&);
    ~A();
  };

  struct B : A {
  };

  void test(B b) {
    extern void consume(void(^)());
    consume(^{ (void) b; });
  }
}

namespace test4 {
  struct A {
    A();
    ~A();
  };

  void foo(A a);

  void test() {
    extern void consume(void(^)());
    consume(^{ return foo(A()); });
  }
  // CHECK-LABEL: define{{.*}} void @_ZN5test44testEv()
  // CHECK-LABEL: define internal void @___ZN5test44testEv_block_invoke
  // CHECK: [[TMP:%.*]] = alloca [[A:%.*]], align 1
  // CHECK-NEXT: store ptr [[BLOCKDESC:%.*]], ptr {{.*}}, align 8
  // CHECK:      call void @_ZN5test41AC1Ev(ptr {{[^,]*}} [[TMP]])
  // CHECK-NEXT: call void @_ZN5test43fooENS_1AE(ptr nofreeobj noundef align 1 dereferenceable(1) [[TMP]])
  // CHECK-NEXT: call void @_ZN5test41AD1Ev(ptr {{[^,]*}} [[TMP]])
  // CHECK-NEXT: ret void
}

namespace test5 {
  struct A {
    unsigned afield;
    A();
    A(const A&);
    ~A();
    void foo() const;
  };

  void doWithBlock(void(^)());

  void test(bool cond) {
    A x;
    void (^b)() = (cond ? ^{ x.foo(); } : (void(^)()) 0);
    doWithBlock(b);
  }

  // CHECK-LABEL:    define{{.*}} void @_ZN5test54testEb(
  // CHECK:      [[COND:%.*]] = alloca i8
  // CHECK-NEXT: [[X:%.*]] = alloca [[A:%.*]], align 4
  // CHECK-NEXT: [[B:%.*]] = alloca ptr, align 8
  // CHECK-NEXT: [[BLOCK:%.*]] = alloca [[BLOCK_T:.*]], align 8
  // CHECK-NEXT: [[COND_CLEANUP_SAVE:%.*]] = alloca ptr, align 8
  // CHECK-NEXT: [[CLEANUP_ACTIVE:%.*]] = alloca i1
  // CHECK-NEXT: [[T0:%.*]] = zext i1
  // CHECK-NEXT: store i8 [[T0]], ptr [[COND]], align 1
  // CHECK-NEXT: call void @_ZN5test51AC1Ev(ptr {{[^,]*}} [[X]])
  // CHECK-NEXT: [[T0:%.*]] = load i8, ptr [[COND]], align 1
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i8 [[T0]], 0
  // CHECK-NEXT: store i1 false, ptr [[CLEANUP_ACTIVE]]
  // CHECK-NEXT: br i1 [[T1]],

  // CHECK-NOT:  br
  // CHECK:      [[CAPTURE:%.*]] = getelementptr inbounds nuw [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT: call void @_ZN5test51AC1ERKS0_(ptr {{[^,]*}} [[CAPTURE]], ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[X]])
  // CHECK-NEXT: store ptr [[CAPTURE]], ptr [[COND_CLEANUP_SAVE]], align 8
  // CHECK-NEXT: store i1 true, ptr [[CLEANUP_ACTIVE]]
  // CHECK-NEXT: br label
  // CHECK:      br label
  // CHECK:      phi
  // CHECK-NEXT: store
  // CHECK-NEXT: load
  // CHECK-NEXT: call void @_ZN5test511doWithBlockEU13block_pointerFvvE(
  // CHECK-NEXT: [[T0:%.*]] = load i1, ptr [[CLEANUP_ACTIVE]]
  // CHECK-NEXT: br i1 [[T0]]
  // CHECK:      [[T3:%.*]] = load ptr, ptr [[COND_CLEANUP_SAVE]], align 8
  // CHECK-NEXT: call void @_ZN5test51AD1Ev(ptr {{[^,]*}} [[T3]])
  // CHECK-NEXT: br label
  // CHECK:      call void @_ZN5test51AD1Ev(ptr {{[^,]*}} [[X]])
  // CHECK-NEXT: ret void
}

namespace test6 {
  struct A {
    A();
    ~A();
  };

  void foo(const A &, void (^)());
  void bar();

  void test() {
    // Make sure that the temporary cleanup isn't somehow captured
    // within the block.
    foo(A(), ^{ bar(); });
    bar();
  }

  // CHECK-LABEL:    define{{.*}} void @_ZN5test64testEv()
  // CHECK:      [[TEMP:%.*]] = alloca [[A:%.*]], align 1
  // CHECK-NEXT: call void @_ZN5test61AC1Ev(ptr {{[^,]*}} [[TEMP]])
  // CHECK-NEXT: call void @_ZN5test63fooERKNS_1AEU13block_pointerFvvE(
  // CHECK-NEXT: call void @_ZN5test61AD1Ev(ptr {{[^,]*}} [[TEMP]])
  // CHECK-NEXT: call void @_ZN5test63barEv()
  // CHECK-NEXT: ret void
}

namespace test7 {
  int f() {
    static int n;
    int *const p = &n;
    return ^{ return *p; }();
  }
}

namespace test8 {
  // failure to capture this after skipping rebuild of the 'this' pointer.
  struct X {
    int x;

    template<typename T>
    int foo() {
      return ^ { return x; }();
    }
  };

  template int X::foo<int>();
}

namespace test9 {
  struct B {
    void *p;
    B();
    B(const B&);
    ~B();
  };

  void use_block(void (^)());
  void use_block_2(void (^)(), const B &a);

  // Ensuring that creating a non-trivial capture copy expression
  // doesn't end up stealing the block registration for the block we
  // just parsed.  That block must have captures or else it won't
  // force registration.  Must occur within a block for some reason.
  void test() {
    B x;
    use_block(^{
        int y;
        use_block_2(^{ (void)y; }, x);
    });
  }
}

namespace test10 {
  // Check that 'v' is included in the copy helper function name to indicate
  // the constructor taking a volatile parameter is called to copy the captured
  // object.

  // CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_8_32c16_ZTSVN6test101BE(
  // CHECK: call void @_ZN6test101BC1ERVKS0_(
  // CHECK-LABEL: define linkonce_odr hidden void @__destroy_helper_block_8_32c16_ZTSVN6test101BE(
  // CHECK: call void @_ZN6test101BD1Ev(

  struct B {
    int a;
    B();
    B(const B &);
    B(const volatile B &);
    ~B();
  };

  void test() {
    volatile B x;
    ^{ (void)x; };
  }
}

namespace test11 {
  // A block in a default member initializer captures the object being
  // initialized, wherever the initializer happens to be emitted.
  struct A {
    int x;
    int func();
    int (^callMember)() = ^{ return func(); };
    int (^useField)() = ^{ return x; };
  };
  template <class T> struct TA {
    int func();
    int (^callMember)() = ^{ return func(); };
  };
  struct B {
    int func();
    int (^nested)() = ^{ return ^{ return func(); }(); };
    int viaLambda = [this] { return ^{ return func(); }(); }();
    void m();
  };
  struct Other {
    int y;
    void m();
    static void s();
  };

  // CHECK-LABEL: define{{.*}} void @_ZN6test112f1Ev()
  // CHECK:       [[T:%.*]] = alloca %"struct.test11::A",
  // CHECK:       [[BLOCK:%.*]] = alloca [[BLOCK_T:<{.*}>]],
  // CHECK:       store ptr @___ZN6test112f1Ev_block_invoke, ptr
  // CHECK:       [[THIS_ADDR:%.*]] = getelementptr inbounds nuw [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT:  store ptr [[T]], ptr [[THIS_ADDR]],
  // CHECK:       store ptr @___ZN6test112f1Ev_block_invoke_2, ptr
  // CHECK:       store ptr [[T]], ptr
  void f1() { A t{}; }

  // CHECK-LABEL: define internal noundef i32 @___ZN6test112f1Ev_block_invoke(
  // CHECK:       [[THIS:%.*]] = load ptr, ptr
  // CHECK-NEXT:  call noundef i32 @_ZN6test111A4funcEv(ptr {{[^,]*}} [[THIS]])

  // CHECK-LABEL: define internal noundef i32 @___ZN6test112f1Ev_block_invoke_2(
  // CHECK:       [[THIS:%.*]] = load ptr, ptr
  // CHECK-NEXT:  [[X:%.*]] = getelementptr inbounds nuw %"struct.test11::A", ptr [[THIS]], i32 0, i32 0
  // CHECK-NEXT:  load i32, ptr [[X]],

  // Not the 'this' of the member function emitting the initializer.
  // CHECK-LABEL: define{{.*}} void @_ZN6test115Other1mEv(ptr {{[^,]*}} %this)
  // CHECK:       [[T:%.*]] = alloca %"struct.test11::A",
  // CHECK:       [[BLOCK:%.*]] = alloca [[BLOCK_T]],
  // CHECK:       [[THIS_ADDR:%.*]] = getelementptr inbounds nuw [[BLOCK_T]], ptr [[BLOCK]], i32 0, i32 5
  // CHECK-NEXT:  store ptr [[T]], ptr [[THIS_ADDR]],
  void Other::m() { A t{}; }

  // CHECK-LABEL: define{{.*}} void @_ZN6test115Other1sEv()
  // CHECK:       [[T:%.*]] = alloca %"struct.test11::A",
  // CHECK:       store ptr [[T]], ptr
  void Other::s() { A t{}; }

  // CHECK-LABEL: define{{.*}} void @_ZN6test112f2Ev()
  // CHECK:       [[T:%.*]] = alloca %"struct.test11::TA",
  // CHECK:       store ptr [[T]], ptr
  void f2() { TA<int> t{}; }

  // CHECK-LABEL: define{{.*}} void @_ZN6test112f3Ev()
  // CHECK:       [[T:%.*]] = alloca %"struct.test11::B",
  // CHECK:       store ptr [[T]], ptr
  void f3() { B t{}; }

  // CHECK-LABEL: define internal noundef i32 @___ZN6test112f3Ev_block_invoke(
  // CHECK:       [[THIS:%.*]] = load ptr, ptr
  // CHECK:       store ptr [[THIS]], ptr

  // CHECK-LABEL: define internal noundef i32 @___ZN6test112f3Ev_block_invoke_2(
  // CHECK:       [[THIS:%.*]] = load ptr, ptr
  // CHECK-NEXT:  call noundef i32 @_ZN6test111B4funcEv(ptr {{[^,]*}} [[THIS]])

  // A block around a lambda that copies '*this' still captures the pointer.
  // CHECK-LABEL: define{{.*}} void @_ZN6test111B1mEv(
  // CHECK:       alloca <{ ptr, i32, i32, ptr, ptr, ptr }>,
  void B::m() { ^{ return [*this]() mutable { return func(); }(); }(); }

  // CHECK-LABEL: define internal void @__cxx_global_var_init()
  // CHECK:       store ptr @_ZN6test112gaE, ptr
  A ga{};
}

// Copy/dispose helper functions and block descriptors of blocks that capture
// objects that are non-external and non-trivial have internal linkage.

// CHECK-LABEL: define internal void @_ZN12_GLOBAL__N_14testEv(
// CHECK: store ptr @[[BLOCK_DESCRIPTOR22]], ptr %{{.*}}, align 8

// CHECK-LABEL: define internal void @__copy_helper_block_8_32c22_ZTSN12_GLOBAL__N_11BE(
// CHECK-LABEL: define internal void @__destroy_helper_block_8_32c22_ZTSN12_GLOBAL__N_11BE(

namespace {
  struct B {
    int a;
    B();
    B(const B &);
    ~B();
  };

  void test() {
    B x;
    ^{ (void)x; };
  }
}

void callTest() {
  test();
}
