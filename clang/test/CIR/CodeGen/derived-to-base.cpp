// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef enum {
  RequestFailed = -2004,
} enumy;

typedef struct {
  const void* samples;
  int cound;
} buffy;

class C1 {
 public:
  virtual ~C1();
  C1(int i);

  struct IE {
    bool supported = false;
    unsigned version = 0;
  };

  struct IEs {
    IE chain;
  };

  static IEs availableIEs;
  class Layer {
   public:
    Layer(int d);
    virtual ~Layer() {}
  };

  virtual enumy SetStuff(enumy e, buffy b);
  virtual enumy Initialize() = 0;
};

class C2 : public C1 {
 public:
  C2(
    void* p,
    int i
  );

  ~C2() override;

  class Layer : public C1::Layer {
   public:
    Layer(int d, const C2* C1);
    virtual ~Layer();

   protected:
    const C2* m_C1;
  };

  virtual enumy SetStuff(enumy e, buffy b) override;
  virtual enumy Initialize() override;
};

class C3 : public C2 {
  struct Layer : public C2::Layer {
   public:
    Layer(int d, const C2* C1);
    void Initialize();
  };

  virtual enumy Initialize() override;
};

void C3::Layer::Initialize() {
  if (m_C1 == nullptr) {
    return;
  }
  if (m_C1->availableIEs.chain.supported) {
  }
}

// CHECK-DAG: !ty_C23A3ALayer = !cir.struct<class "C2::Layer"
// CHECK-DAG: !ty_C33A3ALayer = !cir.struct<struct "C3::Layer"
// CHECK-DAG: !ty_A = !cir.struct<class "A"
// CHECK-DAG: !ty_A2Ebase = !cir.struct<class "A.base"
// CHECK-DAG: !ty_B = !cir.struct<class "B" {!ty_A2Ebase

// CHECK: cir.func @_ZN2C35Layer10InitializeEv

// CHECK:  cir.scope {
// CHECK:    %2 = cir.base_class_addr(%1 : !cir.ptr<!ty_C33A3ALayer> nonnull) [0] -> !cir.ptr<!ty_C23A3ALayer>
// CHECK:    %3 = cir.get_member %2[1] {name = "m_C1"} : !cir.ptr<!ty_C23A3ALayer> -> !cir.ptr<!cir.ptr<!ty_C2_>>
// CHECK:    %4 = cir.load %3 : !cir.ptr<!cir.ptr<!ty_C2_>>, !cir.ptr<!ty_C2_>
// CHECK:    %5 = cir.const #cir.ptr<null> : !cir.ptr<!ty_C2_>
// CHECK:    %6 = cir.cmp(eq, %4, %5) : !cir.ptr<!ty_C2_>, !cir.bool

enumy C3::Initialize() {
  return C2::Initialize();
}

// CHECK: cir.func @_ZN2C310InitializeEv(%arg0: !cir.ptr<!ty_C3_>
// CHECK:     %0 = cir.alloca !cir.ptr<!ty_C3_>, !cir.ptr<!cir.ptr<!ty_C3_>>, ["this", init] {alignment = 8 : i64}

// CHECK:     cir.store %arg0, %0 : !cir.ptr<!ty_C3_>, !cir.ptr<!cir.ptr<!ty_C3_>>
// CHECK:     %2 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_C3_>>, !cir.ptr<!ty_C3_>
// CHECK:     %3 = cir.base_class_addr(%2 : !cir.ptr<!ty_C3_> nonnull) [0] -> !cir.ptr<!ty_C2_>
// CHECK:     %4 = cir.call @_ZN2C210InitializeEv(%3) : (!cir.ptr<!ty_C2_>) -> !s32i

void vcall(C1 &c1) {
  buffy b;
  enumy e;
  c1.SetStuff(e, b);
}

// CHECK: cir.func @_Z5vcallR2C1(%arg0: !cir.ptr<!ty_C1_>
// CHECK:   %0 = cir.alloca !cir.ptr<!ty_C1_>, !cir.ptr<!cir.ptr<!ty_C1_>>, ["c1", init, const] {alignment = 8 : i64}
// CHECK:   %1 = cir.alloca !ty_buffy, !cir.ptr<!ty_buffy>, ["b"] {alignment = 8 : i64}
// CHECK:   %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["e"] {alignment = 4 : i64}
// CHECK:   %3 = cir.alloca !ty_buffy, !cir.ptr<!ty_buffy>, ["agg.tmp0"] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %0 : !cir.ptr<!ty_C1_>, !cir.ptr<!cir.ptr<!ty_C1_>>
// CHECK:   %4 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_C1_>>, !cir.ptr<!ty_C1_>
// CHECK:   %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.call @_ZN5buffyC2ERKS_(%3, %1) : (!cir.ptr<!ty_buffy>, !cir.ptr<!ty_buffy>) -> ()
// CHECK:   %6 = cir.load %3 : !cir.ptr<!ty_buffy>, !ty_buffy
// CHECK:   %7 = cir.cast(bitcast, %4 : !cir.ptr<!ty_C1_>), !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!ty_C1_>, !s32i, !ty_buffy) -> !s32i>>>>
// CHECK:   %8 = cir.load %7 : !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!ty_C1_>, !s32i, !ty_buffy) -> !s32i>>>>, !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!ty_C1_>, !s32i, !ty_buffy) -> !s32i>>>
// CHECK:   %9 = cir.vtable.address_point( %8 : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!ty_C1_>, !s32i, !ty_buffy) -> !s32i>>>, vtable_index = 0, address_point_index = 2) : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!ty_C1_>, !s32i, !ty_buffy) -> !s32i>>>
// CHECK:   %10 = cir.load align(8) %9 : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!ty_C1_>, !s32i, !ty_buffy) -> !s32i>>>, !cir.ptr<!cir.func<(!cir.ptr<!ty_C1_>, !s32i, !ty_buffy) -> !s32i>>
// CHECK:   %11 = cir.call %10(%4, %5, %6) : (!cir.ptr<!cir.func<(!cir.ptr<!ty_C1_>, !s32i, !ty_buffy) -> !s32i>>, !cir.ptr<!ty_C1_>, !s32i, !ty_buffy) -> !s32i
// CHECK:   cir.return
// CHECK: }

class A {
public:
  int a;
  virtual void foo() {a++;}
};

class B : public A {
public:
  int b;
  void foo ()  { static_cast<A>(*this).foo();}
};

// CHECK: cir.func linkonce_odr @_ZN1B3fooEv(%arg0: !cir.ptr<!ty_B>
// CHECK:   %0 = cir.alloca !cir.ptr<!ty_B>, !cir.ptr<!cir.ptr<!ty_B>>, ["this", init] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %0 : !cir.ptr<!ty_B>, !cir.ptr<!cir.ptr<!ty_B>>
// CHECK:   %1 = cir.load deref %0 : !cir.ptr<!cir.ptr<!ty_B>>, !cir.ptr<!ty_B>
// CHECK:   cir.scope {
// CHECK:     %2 = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["ref.tmp0"] {alignment = 8 : i64}
// CHECK:     %3 = cir.base_class_addr(%1 : !cir.ptr<!ty_B> nonnull) [0] -> !cir.ptr<!ty_A>

// Call @A::A(A const&)
// CHECK:     cir.call @_ZN1AC2ERKS_(%2, %3) : (!cir.ptr<!ty_A>, !cir.ptr<!ty_A>) -> ()

// Call @A::foo()
// CHECK:     cir.call @_ZN1A3fooEv(%2) : (!cir.ptr<!ty_A>) -> ()
// CHECK:   }
// CHECK:   cir.return
// CHECK: }

void t() {
  B b;
  b.foo();
}

struct C : public A {
  int& ref;
  C(int& x) : ref(x) {}
};

// CHECK: cir.func @_Z8test_refv()
// CHECK: cir.get_member %2[1] {name = "ref"}
int test_ref() {
  int x = 42;
  C c(x);
  return c.ref;
}

// Multiple base classes, to test non-zero offsets
struct Base1 { int a; };
struct Base2 { int b; };
struct Derived : Base1, Base2 { int c; };
void test_multi_base() {
  Derived d;

  Base2& bref = d; // no null check needed
  // CHECK: %6 = cir.base_class_addr(%0 : !cir.ptr<!ty_Derived> nonnull) [4] -> !cir.ptr<!ty_Base2_>

  Base2* bptr = &d; // has null pointer check
  // CHECK: %7 = cir.base_class_addr(%0 : !cir.ptr<!ty_Derived>) [4] -> !cir.ptr<!ty_Base2_>

  int a = d.a;
  // CHECK: %8 = cir.base_class_addr(%0 : !cir.ptr<!ty_Derived> nonnull) [0] -> !cir.ptr<!ty_Base1_>
  // CHECK: %9 = cir.get_member %8[0] {name = "a"} : !cir.ptr<!ty_Base1_> -> !cir.ptr<!s32i>

  int b = d.b;
  // CHECK: %11 = cir.base_class_addr(%0 : !cir.ptr<!ty_Derived> nonnull) [4] -> !cir.ptr<!ty_Base2_>
  // CHECK: %12 = cir.get_member %11[0] {name = "b"} : !cir.ptr<!ty_Base2_> -> !cir.ptr<!s32i>

  int c = d.c;
  // CHECK: %14 = cir.get_member %0[2] {name = "c"} : !cir.ptr<!ty_Derived> -> !cir.ptr<!s32i>
}
