// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis %s -std=c++11 -O1 -DWITH_DTOR -triple x86_64 -emit-llvm -o - | FileCheck -check-prefix=CHECK-DTOR %s
// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis %s -std=c++11 -O1 -triple x86_64 -emit-llvm -o - | FileCheck -check-prefix=CHECK-NO-DTOR %s

struct A {
  A();
#ifdef WITH_DTOR
  ~A();
#endif
  char a[1024];
  operator bool() const;
};

template <typename T>
void Foo(T &&);

template <typename T>
void Bar(T &&);

template <typename T>
T Baz();

void Test1() {
  // CHECK-DTOR-LABEL: Test1
  // CHECK-DTOR: call void @llvm.lifetime.start.p0i8(i64 1024, i8* nonnull %[[ADDR:[0-9]+]])
  // CHECK-DTOR: call void @_ZN1AC1Ev(%struct.A* nonnull {{[^,]*}} %[[VAR:[^ ]+]])
  // CHECK-DTOR: call void @_Z3FooIRK1AEvOT_
  // CHECK-DTOR: call void @_ZN1AD1Ev(%struct.A* nonnull {{[^,]*}} %[[VAR]])
  // CHECK-DTOR: call void @llvm.lifetime.end.p0i8(i64 1024, i8* nonnull %[[ADDR]])
  // CHECK-DTOR: call void @llvm.lifetime.start.p0i8(i64 1024, i8* nonnull %[[ADDR:[0-9]+]])
  // CHECK-DTOR: call void @_ZN1AC1Ev(%struct.A* nonnull {{[^,]*}} %[[VAR:[^ ]+]])
  // CHECK-DTOR: call void @_Z3FooIRK1AEvOT_
  // CHECK-DTOR: call void @_ZN1AD1Ev(%struct.A* nonnull {{[^,]*}} %[[VAR]])
  // CHECK-DTOR: call void @llvm.lifetime.end.p0i8(i64 1024, i8* nonnull %[[ADDR]])
  // CHECK-DTOR: }

  // CHECK-NO-DTOR-LABEL: Test1
  // CHECK-NO-DTOR: call void @llvm.lifetime.start.p0i8(i64 1024, i8* nonnull %[[ADDR:[0-9]+]])
  // CHECK-NO-DTOR: call void @_ZN1AC1Ev(%struct.A* nonnull {{[^,]*}} %[[VAR:[^ ]+]])
  // CHECK-NO-DTOR: call void @_Z3FooIRK1AEvOT_
  // CHECK-NO-DTOR: call void @llvm.lifetime.end.p0i8(i64 1024, i8* nonnull %[[ADDR]])
  // CHECK-NO-DTOR: call void @llvm.lifetime.start.p0i8(i64 1024, i8* nonnull %[[ADDR:[0-9]+]])
  // CHECK-NO-DTOR: call void @_ZN1AC1Ev(%struct.A* nonnull {{[^,]*}} %[[VAR:[^ ]+]])
  // CHECK-NO-DTOR: call void @_Z3FooIRK1AEvOT_
  // CHECK-NO-DTOR: call void @llvm.lifetime.end.p0i8(i64 1024, i8* nonnull %[[ADDR]])
  // CHECK-NO-DTOR: }
  {
    const A &a = A{};
    Foo(a);
  }
  {
    const A &a = A{};
    Foo(a);
  }
}

void Test2() {
  // CHECK-DTOR-LABEL: Test2
  // CHECK-DTOR: call void @llvm.lifetime.start.p0i8(i64 1024, i8* nonnull %[[ADDR1:[0-9]+]])
  // CHECK-DTOR: call void @_ZN1AC1Ev(%struct.A* nonnull {{[^,]*}} %[[VAR1:[^ ]+]])
  // CHECK-DTOR: call void @_Z3FooIRK1AEvOT_
  // CHECK-DTOR: call void @llvm.lifetime.start.p0i8(i64 1024, i8* nonnull %[[ADDR2:[0-9]+]])
  // CHECK-DTOR: call void @_ZN1AC1Ev(%struct.A* nonnull {{[^,]*}} %[[VAR2:[^ ]+]])
  // CHECK-DTOR: call void @_Z3FooIRK1AEvOT_
  // CHECK-DTOR: call void @_ZN1AD1Ev(%struct.A* nonnull {{[^,]*}} %[[VAR2]])
  // CHECK-DTOR: call void @llvm.lifetime.end.p0i8(i64 1024, i8* nonnull %[[ADDR2]])
  // CHECK-DTOR: call void @_ZN1AD1Ev(%struct.A* nonnull {{[^,]*}} %[[VAR1]])
  // CHECK-DTOR: call void @llvm.lifetime.end.p0i8(i64 1024, i8* nonnull %[[ADDR1]])
  // CHECK-DTOR: }

  // CHECK-NO-DTOR-LABEL: Test2
  // CHECK-NO-DTOR: call void @llvm.lifetime.start.p0i8(i64 1024, i8* nonnull %[[ADDR1:[0-9]+]])
  // CHECK-NO-DTOR: call void @_ZN1AC1Ev(%struct.A* nonnull {{[^,]*}} %[[VAR1:[^ ]+]])
  // CHECK-NO-DTOR: call void @_Z3FooIRK1AEvOT_
  // CHECK-NO-DTOR: call void @llvm.lifetime.start.p0i8(i64 1024, i8* nonnull %[[ADDR2:[0-9]+]])
  // CHECK-NO-DTOR: call void @_ZN1AC1Ev(%struct.A* nonnull {{[^,]*}} %[[VAR2:[^ ]+]])
  // CHECK-NO-DTOR: call void @_Z3FooIRK1AEvOT_
  // CHECK-NO-DTOR: call void @llvm.lifetime.end.p0i8(i64 1024, i8* nonnull %[[ADDR2]])
  // CHECK-NO-DTOR: call void @llvm.lifetime.end.p0i8(i64 1024, i8* nonnull %[[ADDR1]])
  // CHECK-NO-DTOR: }
  const A &a = A{};
  Foo(a);
  const A &b = A{};
  Foo(b);
}

void Test3() {
  // CHECK-DTOR-LABEL: Test3
  // CHECK-DTOR: call void @llvm.lifetime.start
  // CHECK-DTOR: call void @llvm.lifetime.start

  // if.then:
  // CHECK-DTOR: call void @llvm.lifetime.end

  // cleanup:
  // CHECK-DTOR: call void @llvm.lifetime.end

  // cleanup:
  // CHECK-DTOR: call void @llvm.lifetime.end
  // CHECK-DTOR: }
  const A &a = A{};
  if (const A &b = A(a)) {
    Foo(b);
    return;
  }
  Bar(a);
}

void Test4() {
  // CHECK-DTOR-LABEL: Test4
  // CHECK-DTOR: call void @llvm.lifetime.start

  // for.cond.cleanup:
  // CHECK-DTOR: call void @llvm.lifetime.end

  // for.body:
  // CHECK-DTOR: }
  for (const A &a = A{}; a;) {
    Foo(a);
  }
}

int Test5() {
  // CHECK-DTOR-LABEL: Test5
  // CHECK-DTOR: call void @llvm.lifetime.start
  // CHECK-DTOR: call i32 @_Z3BazIiET_v()
  // CHECK-DTOR: store
  // CHECK-DTOR: call void @_Z3FooIRKiEvOT_
  // CHECK-DTOR: load
  // CHECK-DTOR: call void @llvm.lifetime.end
  // CHECK-DTOR: }
  const int &a = Baz<int>();
  Foo(a);
  return a;
}

void Test6() {
  // CHECK-DTOR-LABEL: Test6
  // CHECK-DTOR: call void @llvm.lifetime.start.p0i8(i64 {{[0-9]+}}, i8* nonnull %[[ADDR:[0-9]+]])
  // CHECK-DTOR: call i32 @_Z3BazIiET_v()
  // CHECK-DTOR: store
  // CHECK-DTOR: call void @_Z3FooIiEvOT_
  // CHECK-DTOR: call void @llvm.lifetime.end.p0i8(i64 {{[0-9]+}}, i8* nonnull %[[ADDR]])
  // CHECK-DTOR: call void @llvm.lifetime.start.p0i8(i64 {{[0-9]+}}, i8* nonnull %[[ADDR:[0-9]+]])
  // CHECK-DTOR: call i32 @_Z3BazIiET_v()
  // CHECK-DTOR: store
  // CHECK-DTOR: call void @_Z3FooIiEvOT_
  // CHECK-DTOR: call void @llvm.lifetime.end.p0i8(i64 {{[0-9]+}}, i8* nonnull %[[ADDR]])
  // CHECK-DTOR: }
  Foo(Baz<int>());
  Foo(Baz<int>());
}

void Test7() {
  // CHECK-DTOR-LABEL: Test7
  // CHECK-DTOR: call void @llvm.lifetime.start.p0i8(i64 1024, i8* nonnull %[[ADDR:[0-9]+]])
  // CHECK-DTOR: call void @_Z3BazI1AET_v({{.*}} %[[SLOT:[^ ]+]])
  // CHECK-DTOR: call void @_Z3FooI1AEvOT_({{.*}} %[[SLOT]])
  // CHECK-DTOR: call void @_ZN1AD1Ev(%struct.A* nonnull {{[^,]*}} %[[SLOT]])
  // CHECK-DTOR: call void @llvm.lifetime.end.p0i8(i64 1024, i8* nonnull %[[ADDR]])
  // CHECK-DTOR: call void @llvm.lifetime.start.p0i8(i64 1024, i8* nonnull %[[ADDR:[0-9]+]])
  // CHECK-DTOR: call void @_Z3BazI1AET_v({{.*}} %[[SLOT:[^ ]+]])
  // CHECK-DTOR: call void @_Z3FooI1AEvOT_({{.*}} %[[SLOT]])
  // CHECK-DTOR: call void @_ZN1AD1Ev(%struct.A* nonnull {{[^,]*}} %[[SLOT]])
  // CHECK-DTOR: call void @llvm.lifetime.end.p0i8(i64 1024, i8* nonnull %[[ADDR]])
  // CHECK-DTOR: }
  Foo(Baz<A>());
  Foo(Baz<A>());
}
