// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

class Mother {
public:
 virtual void MotherFoo() {}
 void simple() { }
 virtual void MotherFoo2() {}
};

class Father {
public:
 virtual void FatherFoo() {}
};

class Child : public Mother, public Father {
public:
 void MotherFoo() override {}
};

int main() {
    Mother *b = new Mother();
    b->MotherFoo();
    b->simple();
    Child *c = new Child();
    c->MotherFoo();
    return 0;
}

// CIR: ![[VTypeInfoA:ty_.*]] = !cir.struct<struct  {!cir.ptr<!cir.int<u, 8>>, !cir.ptr<!cir.int<u, 8>>}>
// CIR: ![[VTypeInfoB:ty_.*]] = !cir.struct<struct  {!cir.ptr<!cir.int<u, 8>>, !cir.ptr<!cir.int<u, 8>>, !cir.int<u, 32>, !cir.int<u, 32>, !cir.ptr<!cir.int<u, 8>>, !cir.int<s, 64>, !cir.ptr<!cir.int<u, 8>>, !cir.int<s, 64>}>
// CIR: ![[VTableTypeMother:ty_.*]] = !cir.struct<struct  {!cir.array<!cir.ptr<!cir.int<u, 8>> x 4>}>
// CIR: ![[VTableTypeFather:ty_.*]] = !cir.struct<struct  {!cir.array<!cir.ptr<!cir.int<u, 8>> x 3>}>
// CIR: ![[VTableTypeChild:ty_.*]] = !cir.struct<struct  {!cir.array<!cir.ptr<!cir.int<u, 8>> x 4>, !cir.array<!cir.ptr<!cir.int<u, 8>> x 3>}>
// CIR: !ty_22Father22 = !cir.struct<class "Father" {!cir.ptr<!cir.ptr<!cir.func<!cir.int<u, 32> ()>>>} #cir.record.decl.ast>
// CIR: !ty_22Mother22 = !cir.struct<class "Mother" {!cir.ptr<!cir.ptr<!cir.func<!cir.int<u, 32> ()>>>} #cir.record.decl.ast>
// CIR: !ty_22Child22 = !cir.struct<class "Child" {!cir.struct<class "Mother" {!cir.ptr<!cir.ptr<!cir.func<!cir.int<u, 32> ()>>>} #cir.record.decl.ast>, !cir.struct<class "Father" {!cir.ptr<!cir.ptr<!cir.func<!cir.int<u, 32> ()>>>} #cir.record.decl.ast>} #cir.record.decl.ast>

// CIR: cir.func linkonce_odr @_ZN6MotherC2Ev(%arg0: !cir.ptr<!ty_22Mother22>
// CIR:   %{{[0-9]+}} = cir.vtable.address_point(@_ZTV6Mother, vtable_index = 0, address_point_index = 2) : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_22Mother22>), !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   cir.store %2, %{{[0-9]+}} : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   cir.return
// CIR: }

// LLVM-DAG: define linkonce_odr void @_ZN6MotherC2Ev(ptr %0)
// LLVM-DAG:   store ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV6Mother, i32 0, i32 0, i32 2), ptr %{{[0-9]+}}, align 8
// LLVM-DAG:   ret void
// LLVM-DAG: }

// CIR: cir.func linkonce_odr @_ZN5ChildC2Ev(%arg0: !cir.ptr<!ty_22Child22>
// CIR:   %{{[0-9]+}} = cir.vtable.address_point(@_ZTV5Child, vtable_index = 0, address_point_index = 2) : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>
// CIR:   %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_22Child22>), !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   %{{[0-9]+}} = cir.vtable.address_point(@_ZTV5Child, vtable_index = 1, address_point_index = 2) : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>
// CIR:   %{{[0-9]+}} = cir.const #cir.int<8> : !s64i
// CIR:   %{{[0-9]+}} = cir.ptr_stride(%{{[0-9]+}} : !cir.ptr<!ty_22Child22>, %{{[0-9]+}} : !s64i), !cir.ptr<!ty_22Child22>
// CIR:   %11 = cir.cast(bitcast, %{{[0-9]+}} : !cir.ptr<!ty_22Child22>), !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.func<!u32i ()>>>>
// CIR:   cir.return
// CIR: }

// LLVM-DAG: define linkonce_odr void @_ZN5ChildC2Ev(ptr %0)
// LLVM-DAG:  store ptr getelementptr inbounds ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV5Child, i32 0, i32 0, i32 2), ptr %{{[0-9]+}}, align 8
// LLVM-DAG:  %{{[0-9]+}} = getelementptr %class.Child, ptr %3, i64 8
// LLVM-DAG:  store ptr getelementptr inbounds ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV5Child, i32 0, i32 1, i32 2), ptr %{{[0-9]+}}, align 8
// LLVM-DAG:  ret void
// }

// CIR: cir.func @main() -> !s32i extra(#fn_attr) {

// CIR:   %{{[0-9]+}} = cir.vtable.address_point( %{{[0-9]+}} : !cir.ptr<!cir.ptr<!cir.func<!void (!cir.ptr<!ty_22Mother22>)>>>, vtable_index = 0, address_point_index = 0) : !cir.ptr<!cir.ptr<!cir.func<!void (!cir.ptr<!ty_22Mother22>)>>>

// CIR:   %{{[0-9]+}} = cir.vtable.address_point( %{{[0-9]+}} : !cir.ptr<!cir.ptr<!cir.func<!void (!cir.ptr<!ty_22Child22>)>>>, vtable_index = 0, address_point_index = 0) : !cir.ptr<!cir.ptr<!cir.func<!void (!cir.ptr<!ty_22Child22>)>>>

// CIR: }

//   vtable for Mother
// CIR: cir.global linkonce_odr @_ZTV6Mother = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI6Mother> : !cir.ptr<!u8i>, #cir.global_view<@_ZN6Mother9MotherFooEv> : !cir.ptr<!u8i>, #cir.global_view<@_ZN6Mother10MotherFoo2Ev> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 4>}> : ![[VTableTypeMother]] {alignment = 8 : i64}
// LLVM-DAG: @_ZTV6Mother = linkonce_odr global { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI6Mother, ptr @_ZN6Mother9MotherFooEv, ptr @_ZN6Mother10MotherFoo2Ev] }

//   vtable for __cxxabiv1::__class_type_info
// CIR: cir.global "private" external @_ZTVN10__cxxabiv117__class_type_infoE : !cir.ptr<!cir.ptr<!u8i>>
// LLVM-DAG: @_ZTVN10__cxxabiv117__class_type_infoE = external global ptr

//   typeinfo name for Mother
// CIR: cir.global linkonce_odr @_ZTS6Mother = #cir.const_array<"6Mother" : !cir.array<!s8i x 7>> : !cir.array<!s8i x 7> {alignment = 1 : i64}
// LLVM-DAG: @_ZTS6Mother = linkonce_odr global [7 x i8] c"6Mother"

//   typeinfo for Mother
// CIR: cir.global constant external @_ZTI6Mother = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS6Mother> : !cir.ptr<!u8i>}> : ![[VTypeInfoA]] {alignment = 8 : i64}
// LLVM-DAG: @_ZTI6Mother = constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i32 2), ptr @_ZTS6Mother }

//   vtable for Father
// CIR: cir.global linkonce_odr @_ZTV6Father = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI6Father> : !cir.ptr<!u8i>, #cir.global_view<@_ZN6Father9FatherFooEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 3>}> : ![[VTableTypeFather]] {alignment = 8 : i64}
// LLVM-DAG: @_ZTV6Father = linkonce_odr global { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI6Father, ptr @_ZN6Father9FatherFooEv] }

//   vtable for Child
// CIR: cir.global linkonce_odr @_ZTV5Child = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI5Child> : !cir.ptr<!u8i>, #cir.global_view<@_ZN5Child9MotherFooEv> : !cir.ptr<!u8i>, #cir.global_view<@_ZN6Mother10MotherFoo2Ev> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 4>, #cir.const_array<[#cir.ptr<-8 : i64> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI5Child> : !cir.ptr<!u8i>, #cir.global_view<@_ZN6Father9FatherFooEv> : !cir.ptr<!u8i>]> : !cir.array<!cir.ptr<!u8i> x 3>}> : ![[VTableTypeChild]] {alignment = 8 : i64}
// LLVM-DAG: @_ZTV5Child = linkonce_odr global { [4 x ptr], [3 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI5Child, ptr @_ZN5Child9MotherFooEv, ptr @_ZN6Mother10MotherFoo2Ev], [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr @_ZTI5Child, ptr @_ZN6Father9FatherFooEv] }

//   vtable for __cxxabiv1::__vmi_class_type_info
// CIR: cir.global "private" external @_ZTVN10__cxxabiv121__vmi_class_type_infoE : !cir.ptr<!cir.ptr<!u8i>>
// LLVM-DAG: @_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global ptr

//   typeinfo name for Child
// CIR: cir.global linkonce_odr @_ZTS5Child = #cir.const_array<"5Child" : !cir.array<!s8i x 6>> : !cir.array<!s8i x 6> {alignment = 1 : i64}
// LLVM-DAG: @_ZTS5Child = linkonce_odr global [6 x i8] c"5Child"

//   typeinfo name for Father
// CIR: cir.global linkonce_odr @_ZTS6Father = #cir.const_array<"6Father" : !cir.array<!s8i x 7>> : !cir.array<!s8i x 7> {alignment = 1 : i64}
// LLVM-DAG: @_ZTS6Father = linkonce_odr global [7 x i8] c"6Father"

//   typeinfo for Father
// CIR: cir.global constant external @_ZTI6Father = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv117__class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS6Father> : !cir.ptr<!u8i>}> : !ty_anon_struct {alignment = 8 : i64}
// LLVM-DAG: @_ZTI6Father = constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i32 2), ptr @_ZTS6Father }

//   typeinfo for Child
// CIR: cir.global constant external @_ZTI5Child = #cir.typeinfo<{#cir.global_view<@_ZTVN10__cxxabiv121__vmi_class_type_infoE, [2 : i32]> : !cir.ptr<!u8i>, #cir.global_view<@_ZTS5Child> : !cir.ptr<!u8i>, #cir.int<0> : !u32i, #cir.int<2> : !u32i, #cir.global_view<@_ZTI6Mother> : !cir.ptr<!u8i>, #cir.int<2> : !s64i, #cir.global_view<@_ZTI6Father> : !cir.ptr<!u8i>, #cir.int<2050> : !s64i}> : ![[VTypeInfoB]] {alignment = 8 : i64}
// LLVM-DAG: @_ZTI5Child = constant { ptr, ptr, i32, i32, ptr, i64, ptr, i64 } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i32 2), ptr @_ZTS5Child, i32 0, i32 2, ptr @_ZTI6Mother, i64 2, ptr @_ZTI6Father, i64 2050 }
