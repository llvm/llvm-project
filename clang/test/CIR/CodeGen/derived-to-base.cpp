// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -clangir-disable-emit-cxx-default -emit-cir %s -o %t.cir
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

// CHECK-DAG: !ty_22C23A3ALayer22 = !cir.struct<class "C2::Layer" {!ty_22C13A3ALayer22, !cir.ptr<!ty_22C222>
// CHECK-DAG: !ty_22C33A3ALayer22 = !cir.struct<struct "C3::Layer" {!ty_22C23A3ALayer22

// CHECK: cir.func @_ZN2C35Layer10InitializeEv

// CHECK:  cir.scope {
// CHECK:    %2 = cir.base_class_addr(%1 : cir.ptr <!ty_22C33A3ALayer22>) -> cir.ptr <!ty_22C23A3ALayer22>
// CHECK:    %3 = "cir.struct_element_addr"(%2) <{member_index = 0 : index, member_name = "m_C1"}> : (!cir.ptr<!ty_22C23A3ALayer22>) -> !cir.ptr<!cir.ptr<!ty_22C222>>
// CHECK:    %4 = cir.load %3 : cir.ptr <!cir.ptr<!ty_22C222>>, !cir.ptr<!ty_22C222>
// CHECK:    %5 = cir.const(#cir.null : !cir.ptr<!ty_22C222>) : !cir.ptr<!ty_22C222>
// CHECK:    %6 = cir.cmp(eq, %4, %5) : !cir.ptr<!ty_22C222>, !cir.bool

enumy C3::Initialize() {
  return C2::Initialize();
}

// CHECK: cir.func @_ZN2C310InitializeEv(%arg0: !cir.ptr<!ty_22C322>
// CHECK:     %0 = cir.alloca !cir.ptr<!ty_22C322>, cir.ptr <!cir.ptr<!ty_22C322>>, ["this", init] {alignment = 8 : i64}

// CHECK:     cir.store %arg0, %0 : !cir.ptr<!ty_22C322>, cir.ptr <!cir.ptr<!ty_22C322>>
// CHECK:     %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22C322>>, !cir.ptr<!ty_22C322>
// CHECK:     %3 = cir.base_class_addr(%2 : cir.ptr <!ty_22C322>) -> cir.ptr <!ty_22C222>
// CHECK:     %4 = cir.call @_ZN2C210InitializeEv(%3) : (!cir.ptr<!ty_22C222>) -> !s32i

void vcall(C1 &c1) {
  buffy b;
  enumy e;
  c1.SetStuff(e, b);
}

// CHECK: cir.func @_Z5vcallR2C1(%arg0: !cir.ptr<!ty_22C122>
// CHECK:   %0 = cir.alloca !cir.ptr<!ty_22C122>, cir.ptr <!cir.ptr<!ty_22C122>>, ["c1", init] {alignment = 8 : i64}
// CHECK:   %1 = cir.alloca !ty_22buffy22, cir.ptr <!ty_22buffy22>, ["b"] {alignment = 8 : i64}
// CHECK:   %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["e"] {alignment = 4 : i64}
// CHECK:   %3 = cir.alloca !ty_22buffy22, cir.ptr <!ty_22buffy22>, ["agg.tmp0"] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %0 : !cir.ptr<!ty_22C122>, cir.ptr <!cir.ptr<!ty_22C122>>
// CHECK:   %4 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22C122>>, !cir.ptr<!ty_22C122>
// CHECK:   %5 = cir.load %2 : cir.ptr <!s32i>, !s32i
// CHECK:   cir.call @_ZN5buffyC2ERKS_(%3, %1) : (!cir.ptr<!ty_22buffy22>, !cir.ptr<!ty_22buffy22>) -> ()
// CHECK:   %6 = cir.load %3 : cir.ptr <!ty_22buffy22>, !ty_22buffy22
// CHECK:   %7 = cir.cast(bitcast, %4 : !cir.ptr<!ty_22C122>), !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22C122>, !s32i, !ty_22buffy22)>>>>
// CHECK:   %8 = cir.load %7 : cir.ptr <!cir.ptr<!cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22C122>, !s32i, !ty_22buffy22)>>>>, !cir.ptr<!cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22C122>, !s32i, !ty_22buffy22)>>>
// CHECK:   %9 = cir.vtable.address_point( %8 : !cir.ptr<!cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22C122>, !s32i, !ty_22buffy22)>>>, vtable_index = 0, address_point_index = 2) : cir.ptr <!cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22C122>, !s32i, !ty_22buffy22)>>>
// CHECK:   %10 = cir.load %9 : cir.ptr <!cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22C122>, !s32i, !ty_22buffy22)>>>, !cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22C122>, !s32i, !ty_22buffy22)>>
// CHECK:   %11 = cir.call %10(%4, %5, %6) : (!cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22C122>, !s32i, !ty_22buffy22)>>, !cir.ptr<!ty_22C122>, !s32i, !ty_22buffy22) -> !s32i
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

// CHECK: cir.func linkonce_odr @_ZN1B3fooEv(%arg0: !cir.ptr<!ty_22B22>
// CHECK:   %0 = cir.alloca !cir.ptr<!ty_22B22>, cir.ptr <!cir.ptr<!ty_22B22>>, ["this", init] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %0 : !cir.ptr<!ty_22B22>, cir.ptr <!cir.ptr<!ty_22B22>>
// CHECK:   %1 = cir.load deref %0 : cir.ptr <!cir.ptr<!ty_22B22>>, !cir.ptr<!ty_22B22>
// CHECK:   cir.scope {
// CHECK:     %2 = cir.alloca !ty_22A22, cir.ptr <!ty_22A22>, ["ref.tmp0"] {alignment = 8 : i64}
// CHECK:     %3 = cir.base_class_addr(%1 : cir.ptr <!ty_22B22>) -> cir.ptr <!ty_22A22>

// Call @A::A(A const&)
// CHECK:     cir.call @_ZN1AC2ERKS_(%2, %3) : (!cir.ptr<!ty_22A22>, !cir.ptr<!ty_22A22>) -> ()

// Call @A::foo()
// CHECK:     cir.call @_ZN1A3fooEv(%2) : (!cir.ptr<!ty_22A22>) -> ()
// CHECK:   }
// CHECK:   cir.return
// CHECK: }

void t() {
  B b;
  b.foo();
}
