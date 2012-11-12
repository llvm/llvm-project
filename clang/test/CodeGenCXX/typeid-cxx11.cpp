// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -emit-llvm -std=c++11 -o - | FileCheck %s
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

// CHECK: @_ZN5Test1L5itemsE = internal constant [4 x {{.*}}] [{{.*}} @_ZTIN5Test11AE {{.*}}, {{.*}}, {{.*}} @_ZN5Test19make_implINS_1AEEEPvv }, {{.*}} @_ZTIN5Test11BE {{.*}} @_ZN5Test19make_implINS_1BEEEPvv {{.*}} @_ZTIN5Test11CE {{.*}} @_ZN5Test19make_implINS_1CEEEPvv {{.*}} @_ZTIi {{.*}} @_ZN5Test19make_implIiEEPvv }]
constexpr Item items[] = {
  item<A>("A"), item<B>("B"), item<C>("C"), item<int>("int")
};

// CHECK: @_ZN5Test11xE = constant %"class.std::type_info"* bitcast ({{.*}}* @_ZTIN5Test11AE to %"class.std::type_info"*), align 8
constexpr auto &x = items[0].ti;

// CHECK: @_ZN5Test11yE = constant %"class.std::type_info"* bitcast ({{.*}}* @_ZTIN5Test11BE to %"class.std::type_info"*), align 8
constexpr auto &y = typeid(B{});

}
