// RUN: %clang_cc1 -triple x86_64-pc-linux -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR
// RUN: %clang_cc1 %s -triple=x86_64-pc-linux -fclangir -emit-llvm -o %t-cir.ll
// RUN: FileCheck %s --input-file=%t-cir.ll --check-prefixes=LLVM
// RUN: %clang_cc1 %s -triple=x86_64-pc-linux -emit-llvm -o %t-classic.ll
// RUN: FileCheck %s --input-file=%t-classic.ll --check-prefixes=LLVM
//
// RUN: %clang_cc1 -triple x86_64-pc-linux -std=c++03 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR
// RUN: %clang_cc1 %s -triple=x86_64-pc-linux -std=c++03 -fclangir -emit-llvm -o %t-cir.ll
// RUN: FileCheck %s --input-file=%t-cir.ll --check-prefixes=LLVM
// RUN: %clang_cc1 %s -triple=x86_64-pc-linux -std=c++03 -emit-llvm -o %t-classic.ll
// RUN: FileCheck %s --input-file=%t-classic.ll --check-prefixes=LLVM
//
// RUN: %clang_cc1 -triple x86_64-pc-linux -std=c++11 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR
// RUN: %clang_cc1 %s -triple=x86_64-pc-linux -std=c++11 -fclangir -emit-llvm -o %t-cir.ll
// RUN: FileCheck %s --input-file=%t-cir.ll --check-prefixes=LLVM
// RUN: %clang_cc1 %s -triple=x86_64-pc-linux -std=c++11 -emit-llvm -o %t-classic.ll
// RUN: FileCheck %s --input-file=%t-classic.ll --check-prefixes=LLVM

namespace {
  struct A {
    virtual void f() { }
  };
}

void f() { A b; }

struct B {
  B();
  virtual void f();
};

B::B() { }

struct C : virtual B {
  C();
  virtual void f() { } 
};

C::C() { } 

struct D {
  virtual void f();
};

void D::f() { }

static struct : D { } e;

// Force 'e' to be constructed and therefore have a vtable defined.
void use_e() {
  e.f();
}

// The destructor is the key function.
template<typename T>
struct E {
  virtual ~E();
};

template<typename T> E<T>::~E() { }

// Anchor is the key function
template<>
struct E<char> {
  virtual void anchor();
};

void E<char>::anchor() { }

template struct E<short>;
extern template struct E<int>;

void use_E() {
  E<int> ei;
  (void)ei;
  E<long> el;
  (void)el;
}

// No key function
template<typename T>
struct F {
  virtual void foo() { }
};

// No key function
template<>
struct F<char> {
  virtual void foo() { }
};

template struct F<short>;
extern template struct F<int>;

void use_F() {
  F<char> fc;
  fc.foo();
  F<int> fi;
  fi.foo();
  F<long> fl;
  (void)fl;
}

// B has a key function that is not defined in this translation unit so its vtable
// has external linkage.
// CIR-DAG: cir.global "private" external @_ZTV1B : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @_ZTV1B = external {{.*}}{ [3 x ptr] }, align 8

// C has no key function, so its vtable should have weak_odr linkage
// and hidden visibility.
// CIR-DAG: cir.global "private" linkonce_odr comdat @_ZTV1C = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1C> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1C1fEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 5>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global linkonce_odr comdat @_ZTS1C = #cir.const_array<"1C" : !cir.array<!s8i x 2>, trailing_zeros> : !cir.array<!s8i x 3> {alignment = 1 : i64}
// CIR-DAG: cir.global constant linkonce_odr comdat @_ZTI1C = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv121__vmi_class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS1C> : !cir.ptr<!u8i>, #cir.int<0> : !u32i, #cir.int<1> : !u32i, #cir.global_view<@_ZTI1B> : !cir.ptr<!u8i>, #cir.int<-8189> : !s64i}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global linkonce_odr comdat @_ZTT1C = #cir.const_array<[#cir.global_view<@_ZTV1C, [0 : i32, 4 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTV1C, [0 : i32, 4 : i32]> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 2> {alignment = 8 : i64}
// LLVM-DAG: @_ZTV1C = linkonce_odr {{.*}}{ [5 x ptr] } { [5 x ptr] [ptr null, ptr null, ptr null, ptr @_ZTI1C, ptr @_ZN1C1fEv] }, comdat, align 8
// LLVM-DAG: @_ZTS1C = linkonce_odr {{.*}}[{{[0-9]}} x i8] c"1C\00", comdat, align 1
// LLVM-DAG: @_ZTI1C = linkonce_odr {{.*}}{ ptr, ptr, i32, i32, ptr, i64 } { ptr getelementptr{{.*}}({{.*}}, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 {{.*}}), ptr @_ZTS1C, i32 0, i32 1, ptr @_ZTI1B, i64 -8189 }, comdat
// LLVM-DAG: @_ZTT1C = linkonce_odr {{.*}}[2 x ptr] [ptr getelementptr inbounds{{.*}}({{.*}}, ptr @_ZTV1C, {{.*}}, ptr getelementptr inbounds {{.*}}({{.*}}, ptr @_ZTV1C{{.*}})]

// D has a key function that is defined in this translation unit so its vtable is
// defined in the translation unit.
// CIR-DAG: cir.global "private" external @_ZTV1D = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1D1fEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 3>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global external @_ZTS1D = #cir.const_array<"1D" : !cir.array<!s8i x 2>, trailing_zeros> : !cir.array<!s8i x 3> {alignment = 1 : i64}
// CIR-DAG: cir.global constant external @_ZTI1D = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS1D> : !cir.ptr<!u8i>}> : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @_ZTV1D = {{(unnamed_addr constant|global)}} { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1D, ptr @_ZN1D1fEv] }, align 8
// LLVM-DAG: @_ZTS1D = {{(constant|global)}} [{{[0-9]}} x i8] c"1D\00", align 1
// LLVM-DAG: @_ZTI1D = constant { ptr, ptr } { ptr getelementptr {{.*}}({{.*}}, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 {{[0-9]+}}), ptr @_ZTS1D }, align 8

// E<char> is an explicit specialization with a key function defined
// in this translation unit, so its vtable should have external
// linkage.
// CIR-DAG: cir.global "private" external @_ZTV1EIcE = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1EIcE> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1EIcE6anchorEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 3>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global external @_ZTS1EIcE = #cir.const_array<"1EIcE" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6> {alignment = 1 : i64}
// CIR-DAG: cir.global constant external @_ZTI1EIcE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS1EIcE> : !cir.ptr<!u8i>}> : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @_ZTV1EIcE = {{(unnamed_addr constant|global)}} { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1EIcE, ptr @_ZN1EIcE6anchorEv] }, align 8
// LLVM-DAG: @_ZTS1EIcE = {{(constant|global)}} [6 x i8] c"1EIcE\00", align 1
// LLVM-DAG: @_ZTI1EIcE = constant { ptr, ptr } { ptr getelementptr {{.*}}({{.*}}, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 {{.*}}), ptr @_ZTS1EIcE }, align 8

// E<short> is an explicit template instantiation with a key function
// defined in this translation unit, so its vtable should have
// weak_odr linkage.
// CIR-DAG: cir.global "private" weak_odr comdat @_ZTV1EIsE = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1EIsE> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1EIsED1Ev> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1EIsED0Ev> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 4>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global constant weak_odr comdat @_ZTI1EIsE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS1EIsE> : !cir.ptr<!u8i>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global weak_odr comdat @_ZTS1EIsE = #cir.const_array<"1EIsE" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6> {alignment = 1 : i64}
// LLVM-DAG: @_ZTV1EIsE = weak_odr {{.*}}{ [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1EIsE, ptr @_ZN1EIsED1Ev, ptr @_ZN1EIsED0Ev] }, comdat, align 8
// LLVM-DAG: @_ZTI1EIsE = weak_odr constant { ptr, ptr } { ptr getelementptr {{.*}}({{.*}}, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 {{.*}}), ptr @_ZTS1EIsE }, comdat
// LLVM-DAG: @_ZTS1EIsE = weak_odr {{.*}}[6 x i8] c"1EIsE\00", comdat

// F<short> is an explicit template instantiation without a key
// function, so its vtable should have weak_odr linkage
// CIR-DAG: cir.global "private" weak_odr comdat @_ZTV1FIsE = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1FIsE> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1FIsE3fooEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 3>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global weak_odr comdat @_ZTS1FIsE = #cir.const_array<"1FIsE" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6> {alignment = 1 : i64}
// CIR-DAG: cir.global constant weak_odr comdat @_ZTI1FIsE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS1FIsE> : !cir.ptr<!u8i>}> : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @_ZTV1FIsE = weak_odr {{.*}}{ [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1FIsE, ptr @_ZN1FIsE3fooEv] }, comdat, align 8
// LLVM-DAG: @_ZTS1FIsE = weak_odr {{.*}}[6 x i8] c"1FIsE\00", comdat, align 1
// LLVM-DAG: @_ZTI1FIsE = weak_odr constant { ptr, ptr } { ptr getelementptr {{.*}}({{.*}}, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 {{.*}}), ptr @_ZTS1FIsE }, comdat

// E<long> is an implicit template instantiation with a key function
// defined in this translation unit, so its vtable should have
// linkonce_odr linkage.
// CIR-DAG: cir.global "private" linkonce_odr comdat @_ZTV1EIlE = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1EIlE> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1EIlED1Ev> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1EIlED0Ev> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 4>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global linkonce_odr comdat @_ZTS1EIlE = #cir.const_array<"1EIlE" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6> {alignment = 1 : i64}
// CIR-DAG: cir.global constant linkonce_odr comdat @_ZTI1EIlE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS1EIlE> : !cir.ptr<!u8i>}> : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @_ZTV1EIlE = linkonce_odr {{.*}}{ [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1EIlE, ptr @_ZN1EIlED1Ev, ptr @_ZN1EIlED0Ev] }, comdat, align 8
// LLVM-DAG: @_ZTS1EIlE = linkonce_odr {{.*}}[6 x i8] c"1EIlE\00", comdat, align 1
// LLVM-DAG: @_ZTI1EIlE = linkonce_odr constant { ptr, ptr } { ptr getelementptr {{.*}}({{.*}}, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 {{.*}}), ptr @_ZTS1EIlE }, comdat

// F<long> is an implicit template instantiation with no key function,
// so its vtable should have linkonce_odr linkage.
// CIR-DAG: cir.global "private" linkonce_odr comdat @_ZTV1FIlE = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1FIlE> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1FIlE3fooEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 3>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global linkonce_odr comdat @_ZTS1FIlE = #cir.const_array<"1FIlE" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6> {alignment = 1 : i64}
// CIR-DAG: cir.global constant linkonce_odr comdat @_ZTI1FIlE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS1FIlE> : !cir.ptr<!u8i>}> : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @_ZTV1FIlE = linkonce_odr {{.*}}{ [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1FIlE, ptr @_ZN1FIlE3fooEv] }, comdat, align 8
// LLVM-DAG: @_ZTS1FIlE = linkonce_odr {{.*}}[6 x i8] c"1FIlE\00", comdat, align 1
// LLVM-DAG: @_ZTI1FIlE = linkonce_odr constant { ptr, ptr } { ptr getelementptr {{.*}}({{.*}}, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 {{.*}}), ptr @_ZTS1FIlE }, comdat

// F<int> is an explicit template instantiation declaration without a
// key function, so its vtable should have external linkage.
// CIR-DAG: cir.global "private" external @_ZTV1FIiE : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @_ZTV1FIiE = external {{.*}}{ [3 x ptr] }, align 8

// E<int> is an explicit template instantiation declaration. It has a
// key function is not instantiated, so we know that vtable definition
// will be generated in TU where key function will be defined
// so we can mark it as external (without optimizations) and
// available_externally (with optimizations) because all of the inline
// virtual functions have been emitted.
// CIR-DAG: cir.global "private" external @_ZTV1EIiE : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @_ZTV1EIiE = external {{.*}}{ [4 x ptr] }, align 8

// The anonymous struct for e has no linkage, so the vtable should have
// internal linkage.
// CIR-DAG: cir.global "private" internal dso_local @_ZTV3$_0 = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI3$_0> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1D1fEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 3>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global internal dso_local @_ZTS3$_0 = #cir.const_array<"3$_0" : !cir.array<!s8i x 4>, trailing_zeros> : !cir.array<!s8i x 5> {alignment = 1 : i64}
// CIR-DAG: cir.global constant internal @_ZTI3$_0 = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv120__si_class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS3$_0> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1D> : !cir.ptr<!u8i>}> : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @"_ZTV3$_0" = internal {{.*}}{ [3 x ptr] } { [3 x ptr] [ptr null, ptr @"_ZTI3$_0", ptr @_ZN1D1fEv] }, align 8
// LLVM-DAG: @"_ZTS3$_0" = internal {{.*}}[5 x i8] c"3$_0\00", align 1
// LLVM-DAG: @"_ZTI3$_0" = internal constant { ptr, ptr, ptr } { ptr getelementptr {{.*}}({{.*}}, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 {{.*}}), ptr @"_ZTS3$_0", ptr @_ZTI1D }, align 8

// The A vtable should have internal linkage since it is inside an anonymous 
// namespace.
// CIR-DAG: cir.global "private" internal dso_local @_ZTVN12_GLOBAL__N_11AE = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTIN12_GLOBAL__N_11AE> : !cir.ptr<!u8i>, #cir.global_view<@_ZN12_GLOBAL__N_11A1fEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 3>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global internal dso_local @_ZTSN12_GLOBAL__N_11AE = #cir.const_array<"N12_GLOBAL__N_11AE" : !cir.array<!s8i x 18>, trailing_zeros> : !cir.array<!s8i x 19> {alignment = 1 : i64}
// CIR-DAG: cir.global constant internal @_ZTIN12_GLOBAL__N_11AE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTSN12_GLOBAL__N_11AE> : !cir.ptr<!u8i>}> : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @_ZTVN12_GLOBAL__N_11AE = internal {{.*}}{ [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTIN12_GLOBAL__N_11AE, ptr @_ZN12_GLOBAL__N_11A1fEv] }, align 8
// LLVM-DAG: @_ZTSN12_GLOBAL__N_11AE = internal {{.*}}[19 x i8] c"N12_GLOBAL__N_11AE\00", align 1
// LLVM-DAG: @_ZTIN12_GLOBAL__N_11AE = internal constant { ptr, ptr } { ptr getelementptr {{.*}}({{.*}}, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 {{.*}}), ptr @_ZTSN12_GLOBAL__N_11AE }, align 8

// F<char> is an explicit specialization without a key function, so
// its vtable should have linkonce_odr linkage.
// CIR-DAG: cir.global "private" linkonce_odr comdat @_ZTV1FIcE = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1FIcE> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1FIcE3fooEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 3>}> : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global linkonce_odr comdat @_ZTS1FIcE = #cir.const_array<"1FIcE" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6> {alignment = 1 : i64}
// CIR-DAG: cir.global constant linkonce_odr comdat @_ZTI1FIcE = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS1FIcE> : !cir.ptr<!u8i>}> : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @_ZTV1FIcE = linkonce_odr {{.*}}{ [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1FIcE, ptr @_ZN1FIcE3fooEv] }, comdat, align 8
// LLVM-DAG: @_ZTS1FIcE = linkonce_odr {{.*}}[6 x i8] c"1FIcE\00", comdat, align 1
// LLVM-DAG: @_ZTI1FIcE = linkonce_odr constant { ptr, ptr } { ptr getelementptr {{.*}}({{.*}}, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 {{.*}}), ptr @_ZTS1FIcE }, comdat

// CIR-DAG: cir.global "private" linkonce_odr comdat @_ZTV1GIiE = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1GIiE> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1GIiE2f0Ev> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1GIiE2f1Ev> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 4>}> : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @_ZTV1GIiE = linkonce_odr {{.*}}{ [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1GIiE, ptr @_ZN1GIiE2f0Ev, ptr @_ZN1GIiE2f1Ev] }, comdat, align 8
template <typename T>
class G {
public:
  G() {}
  virtual void f0();
  virtual void f1();
};
template <>
void G<int>::f1() {}
template <typename T>
void G<T>::f0() {}
void G_f0()  { new G<int>(); }

// H<int> has a key function without a body but it's a template instantiation
// so its VTable must be emitted.
// CIR-DAG: cir.global "private" linkonce_odr comdat @_ZTV1HIiE = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI1HIiE> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1HIiED1Ev> : !cir.ptr<!u8i>, #cir.global_view<@_ZN1HIiED0Ev> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 4>}> : !{{.*}}{alignment = 8 : i64}
// LLVM-DAG: @_ZTV1HIiE = linkonce_odr {{.*}}{ [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1HIiE, ptr @_ZN1HIiED1Ev, ptr @_ZN1HIiED0Ev] }, comdat, align 8
template <typename T>
class H {
public:
  virtual ~H();
};

void use_H() {
  H<int> h;
}

// I<int> has an explicit instantiation declaration and needs a VTT and
// construction vtables.

// CIR-DAG: cir.global "private" external @_ZTV1IIiE : !{{.*}}{alignment = 8 : i64}
// CIR-DAG: cir.global "private" external @_ZTT1IIiE : !cir.array<!cir.ptr<!u8i> x 4> {alignment = 8 : i64}
// LLVM-DAG: @_ZTV1IIiE = external {{.*}}{ [5 x ptr] }, align 8
// LLVM-DAG: @_ZTT1IIiE = external {{.*}}[4 x ptr], align 8
struct VBase1 { virtual void f(); }; struct VBase2 : virtual VBase1 {};
template<typename T>
struct I : VBase2 {};
extern template struct I<int>;
I<int> i;
