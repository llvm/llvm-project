// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -clangir-disable-emit-cxx-default -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s


enum class EFMode { Always, Verbose };

class PSEvent {
 public:
  PSEvent(
      EFMode m,
      const char* n);
  ~PSEvent();

 private:
  const char* n;
  EFMode m;
};

void blue() {
  PSEvent p(EFMode::Verbose, __FUNCTION__);
}

class A
{
public:
    A() noexcept {}
    A(const A&) noexcept = default;

    virtual ~A() noexcept;
    virtual const char* quack() const noexcept;
};

class B : public A
{
public:
    virtual ~B() noexcept {}
};

// Class A
// CHECK: ![[ClassA:ty_.*]] = !cir.struct<"class.A", !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>, #cir.recdecl.ast>

// Class B
// CHECK: ![[ClassB:ty_.*]] = !cir.struct<"class.B", ![[ClassA]]>

// CHECK: cir.func @_Z4bluev()
// CHECK:   %0 = cir.alloca !ty_22class2EPSEvent22, cir.ptr <!ty_22class2EPSEvent22>, ["p", init] {alignment = 8 : i64}
// CHECK:   %1 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK:   %2 = cir.get_global @".str" : cir.ptr <!cir.array<!s8i x 5>>
// CHECK:   %3 = cir.cast(array_to_ptrdecay, %2 : !cir.ptr<!cir.array<!s8i x 5>>), !cir.ptr<!s8i>
// CHECK:   cir.call @_ZN7PSEventC1E6EFModePKc(%0, %1, %3) : (!cir.ptr<!ty_22class2EPSEvent22>, !s32i, !cir.ptr<!s8i>) -> ()
// CHECK:   cir.return
// CHECK: }

// @B::~B() #1 declaration
// CHECK:   cir.func private @_ZN1BD2Ev(!cir.ptr<![[ClassB]]>)

// operator delete(void*) declaration
// CHECK:   cir.func private @_ZdlPv(!cir.ptr<!void>)

// B dtor => @B::~B() #2
// Calls dtor #1
// Calls operator delete
//
// CHECK:   cir.func linkonce_odr @_ZN1BD0Ev(%arg0: !cir.ptr<![[ClassB]]>
// CHECK:     %0 = cir.alloca !cir.ptr<![[ClassB]]>, cir.ptr <!cir.ptr<![[ClassB]]>>, ["this", init] {alignment = 8 : i64}
// CHECK:     cir.store %arg0, %0 : !cir.ptr<![[ClassB]]>, cir.ptr <!cir.ptr<![[ClassB]]>>
// CHECK:     %1 = cir.load %0 : cir.ptr <!cir.ptr<![[ClassB]]>>, !cir.ptr<![[ClassB]]>
// CHECK:     cir.call @_ZN1BD2Ev(%1) : (!cir.ptr<![[ClassB]]>) -> ()
// CHECK:     %2 = cir.cast(bitcast, %1 : !cir.ptr<![[ClassB]]>), !cir.ptr<!void>
// CHECK:     cir.call @_ZdlPv(%2) : (!cir.ptr<!void>) -> ()
// CHECK:     cir.return
// CHECK:   }

void foo() { B(); }
