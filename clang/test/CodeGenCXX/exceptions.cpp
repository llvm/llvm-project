// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -emit-llvm -std=c++98 -o - -fcxx-exceptions -fexceptions | FileCheck -check-prefix=CHECK -check-prefix=CHECK98 %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -emit-llvm -std=c++11 -o - -fcxx-exceptions -fexceptions | FileCheck -check-prefix=CHECK -check-prefix=CHECK11 %s

// CHECK: %[[STRUCT_TEST13_A:.*]] = type { i32, i32 }

typedef __typeof(sizeof(0)) size_t;

// Declare the reserved global placement new.
void *operator new(size_t, void*);

// This just shouldn't crash.
namespace test0 {
  struct allocator {
    allocator();
    allocator(const allocator&);
    ~allocator();
  };

  void f();
  void g(bool b, bool c) {
    if (b) {
      if (!c)
        throw allocator();

      return;
    }
    f();
  }
}

namespace test1 {
  struct A { A(int); A(int, int); ~A(); void *p; };

  A *a() {
    // CHECK:    define{{( dso_local)?}} [[A:%.*]]* @_ZN5test11aEv()
    // CHECK:      [[NEW:%.*]] = call noalias nonnull i8* @_Znwm(i64 8)
    // CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[NEW]] to [[A]]*
    // CHECK-NEXT: invoke void @_ZN5test11AC1Ei([[A]]* {{[^,]*}} [[CAST]], i32 5)
    // CHECK:      ret [[A]]* [[CAST]]
    // CHECK:      call void @_ZdlPv(i8* [[NEW]])
    return new A(5);
  }

  A *b() {
    // CHECK:    define{{( dso_local)?}} [[A:%.*]]* @_ZN5test11bEv()
    // CHECK:      [[NEW:%.*]] = call noalias nonnull i8* @_Znwm(i64 8)
    // CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[NEW]] to [[A]]*
    // CHECK-NEXT: [[FOO:%.*]] = invoke i32 @_ZN5test13fooEv()
    // CHECK:      invoke void @_ZN5test11AC1Ei([[A]]* {{[^,]*}} [[CAST]], i32 [[FOO]])
    // CHECK:      ret [[A]]* [[CAST]]
    // CHECK:      call void @_ZdlPv(i8* [[NEW]])
    extern int foo();
    return new A(foo());
  }

  struct B { B(); ~B(); operator int(); int x; };
  B makeB();

  A *c() {
    // CHECK:    define{{( dso_local)?}} [[A:%.*]]* @_ZN5test11cEv()
    // CHECK:      [[ACTIVE:%.*]] = alloca i1
    // CHECK-NEXT: [[NEW:%.*]] = call noalias nonnull i8* @_Znwm(i64 8)
    // CHECK-NEXT: store i1 true, i1* [[ACTIVE]]
    // CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[NEW]] to [[A]]*
    // CHECK-NEXT: invoke void @_ZN5test11BC1Ev([[B:%.*]]* {{[^,]*}} [[T0:%.*]])
    // CHECK:      [[T1:%.*]] = getelementptr inbounds [[B]], [[B]]* [[T0]], i32 0, i32 0
    // CHECK-NEXT: [[T2:%.*]] = load i32, i32* [[T1]], align 4
    // CHECK-NEXT: invoke void @_ZN5test11AC1Ei([[A]]* {{[^,]*}} [[CAST]], i32 [[T2]])
    // CHECK:      store i1 false, i1* [[ACTIVE]]

    // CHECK98-NEXT: invoke void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[T0]])
    // CHECK11-NEXT: call void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[T0]])

    // CHECK:      ret [[A]]* [[CAST]]
    // CHECK:      [[ISACTIVE:%.*]] = load i1, i1* [[ACTIVE]]
    // CHECK-NEXT: br i1 [[ISACTIVE]]
    // CHECK:      call void @_ZdlPv(i8* [[NEW]])
    return new A(B().x);
  }

  //   rdar://11904428
  //   Terminate landing pads should call __cxa_begin_catch first.
  // CHECK98:      define linkonce_odr hidden void @__clang_call_terminate(i8* %0) [[NI_NR_NUW:#[0-9]+]] comdat
  // CHECK98-NEXT:   [[T0:%.*]] = call i8* @__cxa_begin_catch(i8* %0) [[NUW:#[0-9]+]]
  // CHECK98-NEXT:   call void @_ZSt9terminatev() [[NR_NUW:#[0-9]+]]
  // CHECK98-NEXT:   unreachable

  A *d() {
    // CHECK:    define{{( dso_local)?}} [[A:%.*]]* @_ZN5test11dEv()
    // CHECK:      [[ACTIVE:%.*]] = alloca i1
    // CHECK-NEXT: [[NEW:%.*]] = call noalias nonnull i8* @_Znwm(i64 8)
    // CHECK-NEXT: store i1 true, i1* [[ACTIVE]]
    // CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[NEW]] to [[A]]*
    // CHECK-NEXT: invoke void @_ZN5test11BC1Ev([[B:%.*]]* {{[^,]*}} [[T0:%.*]])
    // CHECK:      [[T1:%.*]] = invoke i32 @_ZN5test11BcviEv([[B]]* {{[^,]*}} [[T0]])
    // CHECK:      invoke void @_ZN5test11AC1Ei([[A]]* {{[^,]*}} [[CAST]], i32 [[T1]])
    // CHECK:      store i1 false, i1* [[ACTIVE]]

    // CHECK98-NEXT: invoke void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[T0]])
    // CHECK11-NEXT: call void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[T0]])

    // CHECK:      ret [[A]]* [[CAST]]
    // CHECK:      [[ISACTIVE:%.*]] = load i1, i1* [[ACTIVE]]
    // CHECK-NEXT: br i1 [[ISACTIVE]]
    // CHECK:      call void @_ZdlPv(i8* [[NEW]])
    return new A(B());
  }

  A *e() {
    // CHECK:    define{{( dso_local)?}} [[A:%.*]]* @_ZN5test11eEv()
    // CHECK:      [[ACTIVE:%.*]] = alloca i1
    // CHECK-NEXT: [[NEW:%.*]] = call noalias nonnull i8* @_Znwm(i64 8)
    // CHECK-NEXT: store i1 true, i1* [[ACTIVE]]
    // CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[NEW]] to [[A]]*
    // CHECK-NEXT: invoke void @_ZN5test11BC1Ev([[B:%.*]]* {{[^,]*}} [[T0:%.*]])
    // CHECK:      [[T1:%.*]] = invoke i32 @_ZN5test11BcviEv([[B]]* {{[^,]*}} [[T0]])
    // CHECK:      invoke void @_ZN5test11BC1Ev([[B]]* {{[^,]*}} [[T2:%.*]])
    // CHECK:      [[T3:%.*]] = invoke i32 @_ZN5test11BcviEv([[B]]* {{[^,]*}} [[T2]])
    // CHECK:      invoke void @_ZN5test11AC1Eii([[A]]* {{[^,]*}} [[CAST]], i32 [[T1]], i32 [[T3]])
    // CHECK:      store i1 false, i1* [[ACTIVE]]

    // CHECK98-NEXT: invoke void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[T2]])
    // CHECK11-NEXT: call void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[T2]])

    // CHECK98:      invoke void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[T0]])
    // CHECK11:      call void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[T0]])

    // CHECK:      ret [[A]]* [[CAST]]
    // CHECK:      [[ISACTIVE:%.*]] = load i1, i1* [[ACTIVE]]
    // CHECK-NEXT: br i1 [[ISACTIVE]]
    // CHECK:      call void @_ZdlPv(i8* [[NEW]])
    return new A(B(), B());
  }
  A *f() {
    return new A(makeB().x);
  }
  A *g() {
    return new A(makeB());
  }
  A *h() {
    return new A(makeB(), makeB());
  }

  A *i() {
    // CHECK:    define{{( dso_local)?}} [[A:%.*]]* @_ZN5test11iEv()
    // CHECK:      [[X:%.*]] = alloca [[A]]*, align 8
    // CHECK:      [[ACTIVE:%.*]] = alloca i1
    // CHECK:      [[NEW:%.*]] = call noalias nonnull i8* @_Znwm(i64 8)
    // CHECK-NEXT: store i1 true, i1* [[ACTIVE]]
    // CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[NEW]] to [[A]]*
    // CHECK-NEXT: invoke void @_ZN5test15makeBEv([[B:%.*]]* sret([[B]]) align 4 [[T0:%.*]])
    // CHECK:      [[T1:%.*]] = invoke i32 @_ZN5test11BcviEv([[B]]* {{[^,]*}} [[T0]])
    // CHECK:      invoke void @_ZN5test11AC1Ei([[A]]* {{[^,]*}} [[CAST]], i32 [[T1]])
    // CHECK:      store i1 false, i1* [[ACTIVE]]
    // CHECK-NEXT: store [[A]]* [[CAST]], [[A]]** [[X]], align 8
    // CHECK:      invoke void @_ZN5test15makeBEv([[B:%.*]]* sret([[B]]) align 4 [[T2:%.*]])
    // CHECK:      [[RET:%.*]] = load [[A]]*, [[A]]** [[X]], align 8

    // CHECK98:      invoke void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[T2]])
    // CHECK11:      call void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[T2]])

    // CHECK98:      invoke void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[T0]])
    // CHECK11:      call void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[T0]])

    // CHECK:      ret [[A]]* [[RET]]
    // CHECK:      [[ISACTIVE:%.*]] = load i1, i1* [[ACTIVE]]
    // CHECK-NEXT: br i1 [[ISACTIVE]]
    // CHECK:      call void @_ZdlPv(i8* [[NEW]])
    A *x;
    return (x = new A(makeB()), makeB(), x);
  }
}

namespace test2 {
  struct A {
    A(int); A(int, int); ~A();
    void *p;
    void *operator new(size_t);
    void operator delete(void*, size_t);
  };

  A *a() {
    // CHECK:    define{{( dso_local)?}} [[A:%.*]]* @_ZN5test21aEv()
    // CHECK:      [[NEW:%.*]] = call i8* @_ZN5test21AnwEm(i64 8)
    // CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[NEW]] to [[A]]*
    // CHECK-NEXT: invoke void @_ZN5test21AC1Ei([[A]]* {{[^,]*}} [[CAST]], i32 5)
    // CHECK:      ret [[A]]* [[CAST]]

    // CHECK98:      invoke void @_ZN5test21AdlEPvm(i8* [[NEW]], i64 8)
    // CHECK11:      call void @_ZN5test21AdlEPvm(i8* [[NEW]], i64 8)

    // CHECK98:      call void @__clang_call_terminate(i8* {{%.*}}) [[NR_NUW]]
    return new A(5);
  }
}

namespace test3 {
  struct A {
    A(int); A(int, int); A(const A&); ~A();
    void *p;
    void *operator new(size_t, void*, double);
    void operator delete(void*, void*, double);
  };

  void *foo();
  double bar();
  A makeA(), *makeAPtr();

  A *a() {
    // CHECK:    define{{( dso_local)?}} [[A:%.*]]* @_ZN5test31aEv()
    // CHECK:      [[FOO:%.*]] = call i8* @_ZN5test33fooEv()
    // CHECK:      [[BAR:%.*]] = call double @_ZN5test33barEv()
    // CHECK:      [[NEW:%.*]] = call i8* @_ZN5test31AnwEmPvd(i64 8, i8* [[FOO]], double [[BAR]])
    // CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[NEW]] to [[A]]*
    // CHECK-NEXT: invoke void @_ZN5test31AC1Ei([[A]]* {{[^,]*}} [[CAST]], i32 5)
    // CHECK:      ret [[A]]* [[CAST]]

    // CHECK98:      invoke void @_ZN5test31AdlEPvS1_d(i8* [[NEW]], i8* [[FOO]], double [[BAR]])
    // CHECK11:      call void @_ZN5test31AdlEPvS1_d(i8* [[NEW]], i8* [[FOO]], double [[BAR]])

    // CHECK98:      call void @__clang_call_terminate(i8* {{%.*}}) [[NR_NUW]]
    return new(foo(),bar()) A(5);
  }

  // rdar://problem/8439196
  A *b(bool cond) {

    // CHECK:    define{{( dso_local)?}} [[A:%.*]]* @_ZN5test31bEb(i1 zeroext
    // CHECK:      [[SAVED0:%.*]] = alloca i8*
    // CHECK-NEXT: [[SAVED1:%.*]] = alloca i8*
    // CHECK-NEXT: [[CLEANUPACTIVE:%.*]] = alloca i1

    // CHECK:      [[COND:%.*]] = trunc i8 {{.*}} to i1
    // CHECK-NEXT: store i1 false, i1* [[CLEANUPACTIVE]]
    // CHECK-NEXT: br i1 [[COND]]
    return (cond ?

    // CHECK:      [[FOO:%.*]] = call i8* @_ZN5test33fooEv()
    // CHECK-NEXT: [[NEW:%.*]] = call i8* @_ZN5test31AnwEmPvd(i64 8, i8* [[FOO]], double [[CONST:.*]])
    // CHECK-NEXT: store i8* [[NEW]], i8** [[SAVED0]]
    // CHECK-NEXT: store i8* [[FOO]], i8** [[SAVED1]]
    // CHECK-NEXT: store i1 true, i1* [[CLEANUPACTIVE]]
    // CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[NEW]] to [[A]]*
    // CHECK-NEXT: invoke void @_ZN5test35makeAEv([[A]]* sret([[A]]) align 8 [[CAST]])
    // CHECK: br label
    //   -> cond.end
            new(foo(),10.0) A(makeA()) :

    // CHECK:      [[MAKE:%.*]] = call [[A]]* @_ZN5test38makeAPtrEv()
    // CHECK:      br label
    //   -> cond.end
            makeAPtr());

    // cond.end:
    // CHECK:      [[RESULT:%.*]] = phi [[A]]* {{.*}}[[CAST]]{{.*}}[[MAKE]]
    // CHECK:      ret [[A]]* [[RESULT]]

    // in the EH path:
    // CHECK:      [[ISACTIVE:%.*]] = load i1, i1* [[CLEANUPACTIVE]]
    // CHECK-NEXT: br i1 [[ISACTIVE]]
    // CHECK:      [[V0:%.*]] = load i8*, i8** [[SAVED0]]
    // CHECK-NEXT: [[V1:%.*]] = load i8*, i8** [[SAVED1]]

    // CHECK98-NEXT: invoke void @_ZN5test31AdlEPvS1_d(i8* [[V0]], i8* [[V1]], double [[CONST]])
    // CHECK11-NEXT: call void @_ZN5test31AdlEPvS1_d(i8* [[V0]], i8* [[V1]], double [[CONST]])
  }
}

namespace test4 {
  struct A {
    A(int); A(int, int); ~A();
    void *p;
    void *operator new(size_t, void*, void*);
    void operator delete(void*, size_t, void*, void*); // not a match
  };

  A *a() {
    // CHECK:    define{{( dso_local)?}} [[A:%.*]]* @_ZN5test41aEv()
    // CHECK:      [[FOO:%.*]] = call i8* @_ZN5test43fooEv()
    // CHECK-NEXT: [[BAR:%.*]] = call i8* @_ZN5test43barEv()
    // CHECK-NEXT: [[NEW:%.*]] = call i8* @_ZN5test41AnwEmPvS1_(i64 8, i8* [[FOO]], i8* [[BAR]])
    // CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[NEW]] to [[A]]*
    // CHECK-NEXT: call void @_ZN5test41AC1Ei([[A]]* {{[^,]*}} [[CAST]], i32 5)
    // CHECK-NEXT: ret [[A]]* [[CAST]]
    extern void *foo(), *bar();

    return new(foo(),bar()) A(5);
  }
}

// PR7908
namespace test5 {
  struct T { T(); ~T(); };

  struct A {
    A(const A &x, const T &t = T());
    ~A();
  };

  void foo();

  // CHECK-LABEL:    define{{.*}} void @_ZN5test54testEv()
  // CHECK:      [[EXNSLOT:%.*]] = alloca i8*
  // CHECK-NEXT: [[SELECTORSLOT:%.*]] = alloca i32
  // CHECK-NEXT: [[A:%.*]] = alloca [[A_T:%.*]], align 1
  // CHECK-NEXT: [[T:%.*]] = alloca [[T_T:%.*]], align 1
  // CHECK-NEXT: invoke void @_ZN5test53fooEv()
  // CHECK:      [[EXN:%.*]] = load i8*, i8** [[EXNSLOT]]
  // CHECK-NEXT: [[ADJ:%.*]] = call i8* @__cxa_get_exception_ptr(i8* [[EXN]])
  // CHECK-NEXT: [[SRC:%.*]] = bitcast i8* [[ADJ]] to [[A_T]]*
  // CHECK-NEXT: invoke void @_ZN5test51TC1Ev([[T_T]]* {{[^,]*}} [[T]])
  // CHECK:      invoke void @_ZN5test51AC1ERKS0_RKNS_1TE([[A_T]]* {{[^,]*}} [[A]], [[A_T]]* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[SRC]], [[T_T]]* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[T]])

  // CHECK98:      invoke void @_ZN5test51TD1Ev([[T_T]]* {{[^,]*}} [[T]])
  // CHECK11:      call void @_ZN5test51TD1Ev([[T_T]]* {{[^,]*}} [[T]])

  // CHECK98:      call i8* @__cxa_begin_catch(i8* [[EXN]]) [[NUW]]
  // CHECK98-NEXT: invoke void @_ZN5test51AD1Ev([[A_T]]* {{[^,]*}} [[A]])

  // CHECK:      call void @__cxa_end_catch()
  void test() {
    try {
      foo();
    } catch (A a) {
    }
  }
}

// PR9303: invalid assert on this
namespace test6 {
  bool cond();
  void test() {
    try {
    lbl:
      if (cond()) goto lbl;
    } catch (...) {
    }
  }
}

// PR9298
namespace test7 {
  struct A { A(); ~A(); };
  struct B {
    // The throw() operator means that a bad allocation is signalled
    // with a null return, which means that the initializer is
    // evaluated conditionally.
    static void *operator new(size_t size) throw();
    B(const A&, B*);
    ~B();
  };

  B *test() {
    // CHECK: define{{( dso_local)?}} [[B:%.*]]* @_ZN5test74testEv()
    // CHECK:      [[OUTER_NEW:%.*]] = alloca i1
    // CHECK-NEXT: alloca [[A:%.*]],
    // CHECK-NEXT: alloca i8*
    // CHECK-NEXT: alloca i32
    // CHECK-NEXT: [[OUTER_A:%.*]] = alloca i1
    // CHECK-NEXT: alloca i8*
    // CHECK-NEXT: [[INNER_NEW:%.*]] = alloca i1
    // CHECK-NEXT: alloca [[A]]
    // CHECK-NEXT: [[INNER_A:%.*]] = alloca i1

    // Allocate the outer object.
    // CHECK-NEXT: [[NEW:%.*]] = call i8* @_ZN5test71BnwEm(
    // CHECK-NEXT: icmp eq i8* [[NEW]], null

    // These stores, emitted before the outermost conditional branch,
    // deactivate the temporary cleanups.
    // CHECK-NEXT: store i1 false, i1* [[OUTER_NEW]]
    // CHECK-NEXT: store i1 false, i1* [[OUTER_A]]
    // CHECK-NEXT: store i1 false, i1* [[INNER_NEW]]
    // CHECK-NEXT: store i1 false, i1* [[INNER_A]]
    // CHECK-NEXT: br i1

    // We passed the first null check; activate that cleanup and continue.
    // CHECK:      store i1 true, i1* [[OUTER_NEW]]
    // CHECK-NEXT: bitcast

    // Create the first A temporary and activate that cleanup.
    // CHECK-NEXT: invoke void @_ZN5test71AC1Ev(
    // CHECK:      store i1 true, i1* [[OUTER_A]]

    // Allocate the inner object.
    // CHECK-NEXT: [[NEW:%.*]] = call i8* @_ZN5test71BnwEm(
    // CHECK-NEXT: icmp eq i8* [[NEW]], null
    // CHECK-NEXT: br i1

    // We passed the second null check; save that pointer, activate
    // that cleanup, and continue.
    // CHECK:      store i8* [[NEW]]
    // CHECK-NEXT: store i1 true, i1* [[INNER_NEW]]
    // CHECK-NEXT: bitcast

    // Build the second A temporary and activate that cleanup.
    // CHECK-NEXT: invoke void @_ZN5test71AC1Ev(
    // CHECK:      store i1 true, i1* [[INNER_A]]

    // Build the inner B object and deactivate the inner delete cleanup.
    // CHECK-NEXT: invoke void @_ZN5test71BC1ERKNS_1AEPS0_(
    // CHECK:      store i1 false, i1* [[INNER_NEW]]
    // CHECK:      phi

    // Build the outer B object and deactivate the outer delete cleanup.
    // CHECK-NEXT: invoke void @_ZN5test71BC1ERKNS_1AEPS0_(
    // CHECK:      store i1 false, i1* [[OUTER_NEW]]
    // CHECK:      phi
    // CHECK-NEXT: store [[B]]*

    // Destroy the inner A object.
    // CHECK-NEXT: load i1, i1* [[INNER_A]]
    // CHECK-NEXT: br i1

    // CHECK98:    invoke void @_ZN5test71AD1Ev(
    // CHECK11:    call void @_ZN5test71AD1Ev(

    // Destroy the outer A object.
    // CHECK:      load i1, i1* [[OUTER_A]]
    // CHECK-NEXT: br i1

    // CHECK98:    invoke void @_ZN5test71AD1Ev(
    // CHECK11:    call void @_ZN5test71AD1Ev(

    return new B(A(), new B(A(), 0));
  }
}

// Just don't crash.
namespace test8 {
  struct A {
    // Having both of these is required to trigger the assert we're
    // trying to avoid.
    A(const A&);
    A&operator=(const A&);

    ~A();
  };

  A makeA();
  void test() {
    throw makeA();
  }
  // CHECK-LABEL: define{{.*}} void @_ZN5test84testEv
}

// Make sure we generate the correct code for the delete[] call which
// happens if A::A() throws.  (We were previously calling delete[] on
// a pointer to the first array element, not the pointer returned by new[].)
// PR10870
namespace test9 {
  struct A {
    A();
    ~A();
  };
  A* test() {
    return new A[10];
  }
  // CHECK: define{{.*}} {{%.*}}* @_ZN5test94testEv
  // CHECK: [[TEST9_NEW:%.*]] = call noalias nonnull i8* @_Znam
  // CHECK: call void @_ZdaPv(i8* [[TEST9_NEW]])
}

// In a destructor with a function-try-block, a return statement in a
// catch handler behaves differently from running off the end of the
// catch handler.  PR13102.
namespace test10 {
  extern void cleanup();
  extern bool suppress;

  struct A { ~A(); };
  A::~A() try { cleanup(); } catch (...) { return; }
  // CHECK-LABEL:    define{{.*}} void @_ZN6test101AD1Ev(
  // CHECK:      invoke void @_ZN6test107cleanupEv()
  // CHECK-NOT:  rethrow
  // CHECK:      ret void

  struct B { ~B(); };
  B::~B() try { cleanup(); } catch (...) {}
  // CHECK-LABEL:    define{{.*}} void @_ZN6test101BD1Ev(
  // CHECK:      invoke void @_ZN6test107cleanupEv()
  // CHECK:      call i8* @__cxa_begin_catch
  // CHECK-NEXT: invoke void @__cxa_rethrow()
  // CHECK:      unreachable

  struct C { ~C(); };
  C::~C() try { cleanup(); } catch (...) { if (suppress) return; }
  // CHECK-LABEL:    define{{.*}} void @_ZN6test101CD1Ev(
  // CHECK:      invoke void @_ZN6test107cleanupEv()
  // CHECK:      call i8* @__cxa_begin_catch
  // CHECK-NEXT: load i8, i8* @_ZN6test108suppressE, align 1
  // CHECK-NEXT: trunc
  // CHECK-NEXT: br i1

  // CHECK98:      call void @__cxa_end_catch()
  // CHECK98-NEXT: br label
  // CHECK11:      invoke void @__cxa_end_catch()
  // CHECK11-NEXT: to label

  // CHECK:      invoke void @__cxa_rethrow()
  // CHECK:      unreachable
}

// Ensure that an exception in a constructor destroys
// already-constructed array members.  PR14514
namespace test11 {
  struct A {
    A();
    ~A() {}
  };

  struct C {
    A single;
    A array[2][3];

    C();
  };

  C::C() {
    throw 0;
  }
  // CHECK-LABEL:    define{{.*}} void @_ZN6test111CC2Ev(
  // CHECK:      [[THIS:%.*]] = load [[C:%.*]]*, [[C:%.*]]** {{%.*}}
  //   Construct single.
  // CHECK-NEXT: [[SINGLE:%.*]] = getelementptr inbounds [[C]], [[C]]* [[THIS]], i32 0, i32 0
  // CHECK-NEXT: call void @_ZN6test111AC1Ev([[A:%.*]]* {{[^,]*}} [[SINGLE]])
  //   Construct array.
  // CHECK-NEXT: [[ARRAY:%.*]] = getelementptr inbounds [[C]], [[C]]* [[THIS]], i32 0, i32 1
  // CHECK-NEXT: [[ARRAYBEGIN:%.*]] = getelementptr inbounds [2 x [3 x [[A]]]], [2 x [3 x [[A]]]]* [[ARRAY]], i32 0, i32 0, i32 0
  // CHECK-NEXT: [[ARRAYEND:%.*]] = getelementptr inbounds [[A]], [[A]]* [[ARRAYBEGIN]], i64 6
  // CHECK-NEXT: br label
  // CHECK:      [[CUR:%.*]] = phi [[A]]* [ [[ARRAYBEGIN]], {{%.*}} ], [ [[NEXT:%.*]], {{%.*}} ]
  // CHECK-NEXT: invoke void @_ZN6test111AC1Ev([[A:%.*]]* {{[^,]*}} [[CUR]])
  // CHECK:      [[NEXT]] = getelementptr inbounds [[A]], [[A]]* [[CUR]], i64 1
  // CHECK-NEXT: [[DONE:%.*]] = icmp eq [[A]]* [[NEXT]], [[ARRAYEND]]
  // CHECK-NEXT: br i1 [[DONE]],
  //   throw 0;
  // CHECK:      invoke void @__cxa_throw(
  //   Landing pad 1, from constructor in array-initialization loop:
  // CHECK:      landingpad
  //     - First, destroy already-constructed bits of array.
  // CHECK:      [[EMPTY:%.*]] = icmp eq [[A]]* [[ARRAYBEGIN]], [[CUR]]
  // CHECK-NEXT: br i1 [[EMPTY]]
  // CHECK:      [[AFTER:%.*]] = phi [[A]]* [ [[CUR]], {{%.*}} ], [ [[ELT:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[ELT]] = getelementptr inbounds [[A]], [[A]]* [[AFTER]], i64 -1

  // CHECK98-NEXT: invoke void @_ZN6test111AD1Ev([[A]]* {{[^,]*}} [[ELT]])
  // CHECK11-NEXT: call void @_ZN6test111AD1Ev([[A]]* {{[^,]*}} [[ELT]])

  // CHECK:      [[DONE:%.*]] = icmp eq [[A]]* [[ELT]], [[ARRAYBEGIN]]
  // CHECK-NEXT: br i1 [[DONE]],
  //     - Next, chain to cleanup for single.
  // CHECK:      br label
  //   Landing pad 2, from throw site.
  // CHECK:      landingpad
  //     - First, destroy all of array.
  // CHECK:      [[ARRAYBEGIN:%.*]] = getelementptr inbounds [2 x [3 x [[A]]]], [2 x [3 x [[A]]]]* [[ARRAY]], i32 0, i32 0, i32 0
  // CHECK-NEXT: [[ARRAYEND:%.*]] = getelementptr inbounds [[A]], [[A]]* [[ARRAYBEGIN]], i64 6
  // CHECK-NEXT: br label
  // CHECK:      [[AFTER:%.*]] = phi [[A]]* [ [[ARRAYEND]], {{%.*}} ], [ [[ELT:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[ELT]] = getelementptr inbounds [[A]], [[A]]* [[AFTER]], i64 -1

  // CHECK98-NEXT: invoke void @_ZN6test111AD1Ev([[A]]* {{[^,]*}} [[ELT]])
  // CHECK11-NEXT: call void @_ZN6test111AD1Ev([[A]]* {{[^,]*}} [[ELT]])

  // CHECK:      [[DONE:%.*]] = icmp eq [[A]]* [[ELT]], [[ARRAYBEGIN]]
  // CHECK-NEXT: br i1 [[DONE]],
  //     - Next, chain to cleanup for single.
  // CHECK:      br label
  //   Finally, the cleanup for single.

  // CHECK98:      invoke void @_ZN6test111AD1Ev([[A]]* {{[^,]*}} [[SINGLE]])
  // CHECK11:      call void @_ZN6test111AD1Ev([[A]]* {{[^,]*}} [[SINGLE]])

  // CHECK:      br label
  // CHECK:      resume
  //   (After this is a terminate landingpad.)
}

namespace test12 {
  struct A {
    void operator delete(void *, void *);
    A();
  };

  A *test(void *ptr) {
    return new (ptr) A();
  }
  // CHECK-LABEL: define {{.*}} @_ZN6test124testEPv(
  // CHECK:       [[PTR:%.*]] = load i8*, i8*
  // CHECK-NEXT:  [[CAST:%.*]] = bitcast i8* [[PTR]] to [[A:%.*]]*
  // CHECK-NEXT:  invoke void @_ZN6test121AC1Ev([[A]]* {{[^,]*}} [[CAST]])
  // CHECK:       ret [[A]]* [[CAST]]

  // CHECK98:       invoke void @_ZN6test121AdlEPvS1_(i8* [[PTR]], i8* [[PTR]])
  // CHECK11:       call void @_ZN6test121AdlEPvS1_(i8* [[PTR]], i8* [[PTR]])
}

namespace test13 {

struct A {
  A();
  ~A();
  int a, b;
};

// CHECK: define{{.*}} void @_ZN6test134testEi(
// CHECK: %[[REF_TMP:.*]] = alloca %[[STRUCT_TEST13_A]], align 4
// CHECK: %[[CLEANUP_COND:.*]] = alloca i1, align 1
// CHECK: %[[REF_TMP1:.*]] = alloca %[[STRUCT_TEST13_A]], align 4
// CHECK: %[[CLEANUP_COND2:.*]] = alloca i1, align 1

// CHECK: call void @_ZN6test131AC1Ev(%[[STRUCT_TEST13_A]]* {{[^,]*}} %[[REF_TMP]])
// CHECK: store i1 true, i1* %[[CLEANUP_COND]], align 1
// CHECK: br

// CHECK: invoke void @_ZN6test131AC1Ev(%[[STRUCT_TEST13_A]]* {{[^,]*}} %[[REF_TMP1]])

// CHECK: store i1 true, i1* %[[CLEANUP_COND2]], align 1
// CHECK: br

// Check the flag before destructing the temporary.

// CHECK: landingpad { i8*, i32 }
// CHECK: %[[CLEANUP_IS_ACTIVE:.*]] = load i1, i1* %[[CLEANUP_COND]], align 1
// CHECK: br i1 %[[CLEANUP_IS_ACTIVE]],

// CHECK: void @_ZN6test131AD1Ev(%[[STRUCT_TEST13_A]]* {{[^,]*}} %[[REF_TMP]])

void test(int c) {
  const A &s = c ? static_cast<const A &>(A()) : static_cast<const A &>(A());
}

}

// CHECK98: attributes [[NI_NR_NUW]] = { noinline noreturn nounwind }
