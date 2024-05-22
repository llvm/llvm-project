// RUN: %clang_cc1 -I%S %s -triple amdgcn-amd-amdhsa -emit-llvm -std=c++11 -o - | FileCheck %s
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

// CHECK: @_ZN5Test15itemsE ={{.*}} constant [4 x {{.*}}] [{{.*}} ptr addrspacecast (ptr addrspace(1) @_ZTIN5Test11AE to ptr), {{.*}} @_ZN5Test19make_implINS_1AEEEPvv {{.*}} ptr addrspacecast (ptr addrspace(1) @_ZTIN5Test11BE to ptr), {{.*}} @_ZN5Test19make_implINS_1BEEEPvv {{.*}} ptr addrspacecast (ptr addrspace(1) @_ZTIN5Test11CE to ptr), {{.*}} @_ZN5Test19make_implINS_1CEEEPvv {{.*}} ptr addrspacecast (ptr addrspace(1) @_ZTIi to ptr), {{.*}} @_ZN5Test19make_implIiEEPvv }]
extern constexpr Item items[] = {
  item<A>("A"), item<B>("B"), item<C>("C"), item<int>("int")
};

// CHECK: @_ZN5Test11xE ={{.*}} constant ptr addrspacecast (ptr addrspace(1) @_ZTIN5Test11AE to ptr), align 8
constexpr auto &x = items[0].ti;

// CHECK: @_ZN5Test11yE ={{.*}} constant ptr addrspacecast (ptr addrspace(1) @_ZTIN5Test11BE to ptr), align 8
constexpr auto &y = typeid(B{});

}
