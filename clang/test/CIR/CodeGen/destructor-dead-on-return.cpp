// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-lifetime-dse -emit-cir %s -o %t-nodse.cir
// RUN: FileCheck --check-prefix=NODSE --input-file=%t-nodse.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-lifetime-dse -emit-llvm %s -o %t-nodse-cir.ll
// RUN: FileCheck --check-prefix=NODSE --input-file=%t-nodse-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-lifetime-dse -emit-llvm %s -o %t-nodse.ll
// RUN: FileCheck --check-prefix=NODSE --input-file=%t-nodse.ll %s

// A destructor's `this` is dead_on_return for the bytes of the object's data,
// unless lifetime DSE is disabled.

// NODSE-NOT: dead_on_return

void opaque();
void opaque(char *);

class Foo {
public:
  ~Foo();

private:
  char fooVar[4];
};

Foo::~Foo() { opaque(fooVar); }

// CIR: cir.func {{.*}} @_ZN3FooD2Ev(%arg0: !cir.ptr<!rec_Foo> {llvm.align = 1 : i64, llvm.dead_on_return = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef}
// CIR: cir.func {{.*}} @_ZN3FooD1Ev(%arg0: !cir.ptr<!rec_Foo> {llvm.align = 1 : i64, llvm.dead_on_return = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef}
// CIR:   cir.call @_ZN3FooD2Ev(%{{.*}}) nothrow : (!cir.ptr<!rec_Foo> {llvm.align = 1 : i64, llvm.dead_on_return = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef}) -> ()

// LLVM: define dso_local void @_ZN3FooD2Ev(ptr noundef nonnull align 1 dead_on_return(4) dereferenceable(4) %{{.*}})
// LLVM: define dso_local void @_ZN3FooD1Ev(ptr noundef nonnull align 1 dead_on_return(4) dereferenceable(4) %{{.*}})
// LLVM:   call void @_ZN3FooD2Ev(ptr noundef nonnull align 1 dead_on_return(4) dereferenceable(4) %{{.*}})

// The data of a derived class includes its bases.
class BarInheritsFoo : public Foo {
public:
  ~BarInheritsFoo();

private:
  char barVar[12];
};

BarInheritsFoo::~BarInheritsFoo() { opaque(barVar); }

// CIR: cir.func {{.*}} @_ZN14BarInheritsFooD2Ev(%arg0: !cir.ptr<!rec_BarInheritsFoo> {llvm.align = 1 : i64, llvm.dead_on_return = 16 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}
// CIR:   cir.call @_ZN3FooD2Ev(%{{.*}}) nothrow : (!cir.ptr<!rec_Foo> {llvm.align = 1 : i64, llvm.dead_on_return = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef}) -> ()

// LLVM: define dso_local void @_ZN14BarInheritsFooD2Ev(ptr noundef nonnull align 1 dead_on_return(16) dereferenceable(16) %{{.*}})
// LLVM:   call void @_ZN3FooD2Ev(ptr noundef nonnull align 1 dead_on_return(4) dereferenceable(4) %{{.*}})

// A virtual base outlives the destructor of the object holding it.
class BarVirtualInheritsFoo : public virtual Foo {
public:
  ~BarVirtualInheritsFoo();

private:
  char barVar[16];
};

BarVirtualInheritsFoo::~BarVirtualInheritsFoo() { opaque(barVar); }

// CIR: cir.func {{.*}} @_ZN21BarVirtualInheritsFooD2Ev(%arg0: !cir.ptr<!rec_BarVirtualInheritsFoo> {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}{{.*}}, %arg1: !cir.ptr<!cir.ptr<!void>> {llvm.noundef}
// CIR: cir.func {{.*}} @_ZN21BarVirtualInheritsFooD1Ev(%arg0: !cir.ptr<!rec_BarVirtualInheritsFoo> {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}

// LLVM: define dso_local void @_ZN21BarVirtualInheritsFooD2Ev(ptr noundef nonnull align 8 dereferenceable(24) %{{.*}}, ptr noundef %{{.*}})
// LLVM: define dso_local void @_ZN21BarVirtualInheritsFooD1Ev(ptr noundef nonnull align 8 dereferenceable(24) %{{.*}})

// An object with no data gets no attribute.
class Empty {
public:
  ~Empty();
};

Empty::~Empty() { opaque(); }

// CIR: cir.func {{.*}} @_ZN5EmptyD2Ev(%arg0: !cir.ptr<!rec_Empty> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}

// LLVM: define dso_local void @_ZN5EmptyD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %{{.*}})

// The deleting destructor's operator delete may still use the storage.
class Poly {
public:
  virtual ~Poly();

private:
  char polyVar[4];
};

Poly::~Poly() { opaque(polyVar); }

// CIR: cir.func {{.*}} @_ZN4PolyD2Ev(%arg0: !cir.ptr<!rec_Poly> {llvm.align = 8 : i64, llvm.dead_on_return = 12 : i64, llvm.dereferenceable = 12 : i64, llvm.nonnull, llvm.noundef}
// CIR: cir.func {{.*}} @_ZN4PolyD0Ev(%arg0: !cir.ptr<!rec_Poly> {llvm.align = 8 : i64, llvm.dereferenceable = 12 : i64, llvm.nonnull, llvm.noundef}
// CIR:   cir.call @_ZN4PolyD1Ev(%{{.*}}) nothrow : (!cir.ptr<!rec_Poly> {llvm.align = 8 : i64, llvm.dead_on_return = 12 : i64, llvm.dereferenceable = 12 : i64, llvm.nonnull, llvm.noundef}) -> ()

// LLVM: define dso_local void @_ZN4PolyD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %{{.*}})
// LLVM: define dso_local void @_ZN4PolyD0Ev(ptr noundef nonnull align 8 dereferenceable(12) %{{.*}})
// LLVM:   call void @_ZN4PolyD1Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %{{.*}})

void callDtor() {
  Foo f;
}

// CIR: cir.func {{.*}} @_Z8callDtorv()
// CIR:   cir.call @_ZN3FooD1Ev(%{{.*}}) nothrow : (!cir.ptr<!rec_Foo> {llvm.align = 1 : i64, llvm.dead_on_return = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef}) -> ()

// LLVM: define dso_local void @_Z8callDtorv()
// LLVM:   call void @_ZN3FooD1Ev(ptr noundef nonnull align 1 dead_on_return(4) dereferenceable(4) %{{.*}})

void deletePoly(Poly *p) { delete p; }

// CIR: cir.func {{.*}} @_Z10deletePolyP4Poly
// CIR:   cir.call %{{.*}}(%{{.*}}) nothrow : (!cir.ptr<!cir.func<(!cir.ptr<!rec_Poly>)>>, !cir.ptr<!rec_Poly> {llvm.align = 8 : i64, llvm.dereferenceable = 12 : i64, llvm.nonnull, llvm.noundef}) -> ()

// LLVM: define dso_local void @_Z10deletePolyP4Poly
// LLVM:   call void %{{.*}}(ptr noundef nonnull align 8 dereferenceable(12) %{{.*}})
