// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -clangir-disable-emit-cxx-default -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

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
// CHECK: ![[ClassA:ty_.*]] = !cir.struct<"class.A", !cir.ptr<!cir.ptr<() -> i32>>, #cir.recdecl.ast>

// Class B
// CHECK: ![[ClassB:ty_.*]] = !cir.struct<"class.B", ![[ClassA]]>

// @B::~B() #1 declaration
// CHECK:   cir.func @_ZN1BD2Ev(!cir.ptr<![[ClassB]]>) attributes {sym_visibility = "private"}

// operator delete(void*) declaration
// CHECK:   cir.func @_ZdlPv(!cir.ptr<i8>) attributes {sym_visibility = "private"}

// B dtor => @B::~B() #2
// Calls dtor #1
// Calls operator delete
//
// CHECK:   cir.func linkonce_odr @_ZN1BD0Ev(%arg0: !cir.ptr<![[ClassB]]>
// CHECK:     %0 = cir.alloca !cir.ptr<![[ClassB]]>, cir.ptr <!cir.ptr<![[ClassB]]>>, ["this", init] {alignment = 8 : i64}
// CHECK:     cir.store %arg0, %0 : !cir.ptr<![[ClassB]]>, cir.ptr <!cir.ptr<![[ClassB]]>>
// CHECK:     %1 = cir.load %0 : cir.ptr <!cir.ptr<![[ClassB]]>>, !cir.ptr<![[ClassB]]>
// CHECK:     cir.call @_ZN1BD2Ev(%1) : (!cir.ptr<![[ClassB]]>) -> ()
// CHECK:     %2 = cir.cast(bitcast, %1 : !cir.ptr<![[ClassB]]>), !cir.ptr<i8>
// CHECK:     cir.call @_ZdlPv(%2) : (!cir.ptr<i8>) -> ()
// CHECK:     cir.return
// CHECK:   }

void foo() { B(); }
