// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR
// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-apple-darwin10 -fclangir -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-apple-darwin10 -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=LLVM

#include <typeinfo>

namespace Test1 {

struct Item {
  const std::type_info &ti;
  const char *name;
  void *(*make)();
};

template<typename T> void *make_impl() { return new T; }
template<typename T> constexpr Item item(const char *name) {
  return { typeid(T), name, make_impl<T> };
}

struct A { virtual ~A(); };
struct B : virtual A {};
struct C { int n; };

// CIR: cir.global constant external @_ZN5Test15itemsE = #cir.const_array<[
// CIR-SAME: #cir.const_record<{#cir.global_view<@_ZTIN5Test11AE> : !cir.ptr<!rec_std3A3Atype_info>, #cir.global_view<@".str"> : !cir.ptr<!s8i>, #cir.global_view<@_ZN5Test19make_implINS_1AEEEPvv> : !cir.ptr<!cir.func<() -> !cir.ptr<!void>>>}> : !rec_Test13A3AItem
// CIR-SAME: #cir.const_record<{#cir.global_view<@_ZTIN5Test11BE> : !cir.ptr<!rec_std3A3Atype_info>, #cir.global_view<@".str.1"> : !cir.ptr<!s8i>, #cir.global_view<@_ZN5Test19make_implINS_1BEEEPvv> : !cir.ptr<!cir.func<() -> !cir.ptr<!void>>>}> : !rec_Test13A3AItem
// CIR-SAME: #cir.const_record<{#cir.global_view<@_ZTIN5Test11CE> : !cir.ptr<!rec_std3A3Atype_info>, #cir.global_view<@".str.2"> : !cir.ptr<!s8i>, #cir.global_view<@_ZN5Test19make_implINS_1CEEEPvv> : !cir.ptr<!cir.func<() -> !cir.ptr<!void>>>}> : !rec_Test13A3AItem
// CIR-SAME: #cir.const_record<{#cir.global_view<@_ZTIi> : !cir.ptr<!rec_std3A3Atype_info>, #cir.global_view<@".str.3"> : !cir.ptr<!s8i>, #cir.global_view<@_ZN5Test19make_implIiEEPvv> : !cir.ptr<!cir.func<() -> !cir.ptr<!void>>>}> : !rec_Test13A3AItem
// CIR-SAME: ]> : !cir.array<!rec_Test13A3AItem x 4>
//
// LLVM: @_ZN5Test15itemsE ={{.*}} constant [4 x {{.*}}] [{{.*}} @_ZTIN5Test11AE, {{.*}} @_ZN5Test19make_implINS_1AEEEPvv {{.*}} @_ZTIN5Test11BE, {{.*}} @_ZN5Test19make_implINS_1BEEEPvv {{.*}} @_ZTIN5Test11CE, {{.*}} @_ZN5Test19make_implINS_1CEEEPvv {{.*}} @_ZTIi, {{.*}} @_ZN5Test19make_implIiEEPvv }]
extern constexpr Item items[] = {
  item<A>("A"), item<B>("B"), item<C>("C"), item<int>("int")
};

// CIR: cir.global constant external @_ZN5Test11xE = #cir.global_view<@_ZTIN5Test11AE>
// LLVM: @_ZN5Test11xE ={{.*}} constant ptr @_ZTIN5Test11AE, align 8
constexpr auto &x = items[0].ti;

// CIR: cir.global constant external @_ZN5Test11yE = #cir.global_view<@_ZTIN5Test11BE>
// LLVM: @_ZN5Test11yE ={{.*}} constant ptr @_ZTIN5Test11BE, align 8
constexpr auto &y = typeid(B{});

}
