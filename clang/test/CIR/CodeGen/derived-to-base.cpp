// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -clangir-disable-emit-cxx-default -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef enum {
  RequestFailed = -2004,
} enumy;

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

// CHECK: !ty_22class2EC23A3ALayer22 = !cir.struct<"class.C2::Layer", !ty_22class2EC13A3ALayer22, !cir.ptr<!ty_22class2EC222>
// CHECK: !ty_22struct2EC33A3ALayer22 = !cir.struct<"struct.C3::Layer", !ty_22class2EC23A3ALayer22

// CHECK: cir.func @_ZN2C35Layer10InitializeEv

// CHECK:  cir.scope {
// CHECK:    %2 = cir.base_class_addr(%1 : cir.ptr <!ty_22struct2EC33A3ALayer22>) -> cir.ptr <!ty_22class2EC23A3ALayer22>
// CHECK:    %3 = "cir.struct_element_addr"(%2) <{member_name = "m_C1"}> : (!cir.ptr<!ty_22class2EC23A3ALayer22>) -> !cir.ptr<!cir.ptr<!ty_22class2EC222>>
// CHECK:    %4 = cir.load %3 : cir.ptr <!cir.ptr<!ty_22class2EC222>>, !cir.ptr<!ty_22class2EC222>
// CHECK:    %5 = cir.const(#cir.null : !cir.ptr<!ty_22class2EC222>) : !cir.ptr<!ty_22class2EC222>
// CHECK:    %6 = cir.cmp(eq, %4, %5) : !cir.ptr<!ty_22class2EC222>, !cir.bool

enumy C3::Initialize() {
  return C2::Initialize();
}

// CHECK: cir.func @_ZN2C310InitializeEv(%arg0: !cir.ptr<!ty_22class2EC322>
// CHECK:     %0 = cir.alloca !cir.ptr<!ty_22class2EC322>, cir.ptr <!cir.ptr<!ty_22class2EC322>>, ["this", init] {alignment = 8 : i64}

// CHECK:     cir.store %arg0, %0 : !cir.ptr<!ty_22class2EC322>, cir.ptr <!cir.ptr<!ty_22class2EC322>>
// CHECK:     %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22class2EC322>>, !cir.ptr<!ty_22class2EC322>
// CHECK:     %3 = cir.base_class_addr(%2 : cir.ptr <!ty_22class2EC322>) -> cir.ptr <!ty_22class2EC222>
// CHECK:     %4 = cir.call @_ZN2C210InitializeEv(%3) : (!cir.ptr<!ty_22class2EC222>) -> !s32i
