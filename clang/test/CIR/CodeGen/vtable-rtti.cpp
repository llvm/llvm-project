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

// vtable for A type
// CHECK: ![[VTableTypeA:ty_.*]] = !cir.struct<"", !cir.array<!cir.ptr<i8> x 5>>

// Type info B.
// CHECK: ![[TypeInfoB:ty_.*]] = !cir.struct<"", !cir.ptr<i8>, !cir.ptr<i8>, !cir.ptr<i8>>

// Class A
// CHECK: ![[ClassA:ty_.*]] = !cir.struct<"class.A", !cir.ptr<!cir.ptr<() -> i32>>, #cir.recdecl.ast>

// Class B
// CHECK: ![[ClassB:ty_.*]] = !cir.struct<"class.B", ![[ClassA]]>

// B ctor => @B::B()
// Calls @A::A() and initialize __vptr with address of B's vtable.
//
// CHECK: cir.func linkonce_odr @_ZN1BC2Ev(%arg0: !cir.ptr<![[ClassB]]>
// CHECK:   %0 = cir.alloca !cir.ptr<![[ClassB]]>, cir.ptr <!cir.ptr<![[ClassB]]>>, ["this", init] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %0 : !cir.ptr<![[ClassB]]>, cir.ptr <!cir.ptr<![[ClassB]]>>
// CHECK:   %1 = cir.load %0 : cir.ptr <!cir.ptr<![[ClassB]]>>, !cir.ptr<![[ClassB]]>
// CHECK:   %2 = cir.cast(bitcast, %1 : !cir.ptr<![[ClassB]]>), !cir.ptr<![[ClassA]]>
// CHECK:   cir.call @_ZN1AC2Ev(%2) : (!cir.ptr<![[ClassA]]>) -> ()
// CHECK:   %3 = cir.vtable.address_point(@_ZTV1B, vtable_index = 0, address_point_index = 2) : cir.ptr <!cir.ptr<() -> i32>>
// CHECK:   %4 = cir.cast(bitcast, %1 : !cir.ptr<![[ClassB]]>), !cir.ptr<!cir.ptr<!cir.ptr<() -> i32>>>
// CHECK:   cir.store %3, %4 : !cir.ptr<!cir.ptr<() -> i32>>, cir.ptr <!cir.ptr<!cir.ptr<() -> i32>>>
// CHECK:   cir.return
// CHECK: }

// foo - zero initialize object B and call ctor (@B::B())
//
// CHECK: cir.func @_Z3foov() {
// CHECK:     %0 = cir.alloca ![[ClassB]], cir.ptr <![[ClassB]]>, ["agg.tmp0"] {alignment = 8 : i64}
// CHECK:     cir.scope {
// CHECK:       %1 = cir.const(#cir.zero : ![[ClassB]]) : ![[ClassB]]
// CHECK:       cir.store %1, %0 : ![[ClassB]], cir.ptr <![[ClassB]]>
// CHECK:       cir.call @_ZN1BC2Ev(%0) : (!cir.ptr<![[ClassB]]>) -> ()
// CHECK:     }
// CHECK:     cir.return
// CHECK:   }

// Vtable definition for A
// cir.global "private" external @_ZTV1A : ![[VTableTypeA]] {alignment = 8 : i64}

// A ctor => @A::A()
// Calls @A::A() and initialize __vptr with address of A's vtable
//
// CHECK:  cir.func linkonce_odr @_ZN1AC2Ev(%arg0: !cir.ptr<![[ClassA]]>
// CHECK:    %0 = cir.alloca !cir.ptr<![[ClassA]]>, cir.ptr <!cir.ptr<![[ClassA]]>>, ["this", init] {alignment = 8 : i64}
// CHECK:    cir.store %arg0, %0 : !cir.ptr<![[ClassA]]>, cir.ptr <!cir.ptr<![[ClassA]]>>
// CHECK:    %1 = cir.load %0 : cir.ptr <!cir.ptr<![[ClassA]]>>, !cir.ptr<![[ClassA]]>
// CHECK:    %2 = cir.vtable.address_point(@_ZTV1A, vtable_index = 0, address_point_index = 2) : cir.ptr <!cir.ptr<() -> i32>>
// CHECK:    %3 = cir.cast(bitcast, %1 : !cir.ptr<![[ClassA]]>), !cir.ptr<!cir.ptr<!cir.ptr<() -> i32>>>
// CHECK:    cir.store %2, %3 : !cir.ptr<!cir.ptr<() -> i32>>, cir.ptr <!cir.ptr<!cir.ptr<() -> i32>>>
// CHECK:    cir.return
// CHECK:  }

// vtable for B
// CHECK: cir.global linkonce_odr @_ZTV1B = #cir.const_struct<[#cir.const_array<[#cir.null : !cir.ptr<i8>, #cir.global_view<@_ZTI1B> : !cir.ptr<i8>, #cir.global_view<@_ZN1BD1Ev> : !cir.ptr<i8>, #cir.global_view<@_ZN1BD0Ev> : !cir.ptr<i8>, #cir.global_view<@_ZNK1A5quackEv> : !cir.ptr<i8>] : !cir.array<!cir.ptr<i8> x 5>> : !cir.array<!cir.ptr<i8> x 5>]> : ![[VTableTypeA]] {alignment = 8 : i64}

// vtable for __cxxabiv1::__si_class_type_info
// CHECK:   cir.global "private" external @_ZTVN10__cxxabiv120__si_class_type_infoE : !cir.ptr<!cir.ptr<i8>>

// typeinfo name for B
// CHECK:   cir.global linkonce_odr @_ZTS1B = #cir.const_array<"1B" : !cir.array<i8 x 2>> : !cir.array<i8 x 2> {alignment = 1 : i64}

// typeinfo for A
// CHECK:   cir.global "private" constant external @_ZTI1A : !cir.ptr<i8>

// typeinfo for B
// CHECK: cir.global constant external @_ZTI1B = #cir.typeinfo<[#cir.global_view<@_ZTVN10__cxxabiv120__si_class_type_infoE, [2]> : !cir.ptr<i8>, #cir.global_view<@_ZTS1B> : !cir.ptr<i8>, #cir.global_view<@_ZTI1A> : !cir.ptr<i8>] : ![[TypeInfoB]]> : ![[TypeInfoB]] {alignment = 8 : i64}

// Checks for dtors in dtors.cpp

void foo() { B(); }
