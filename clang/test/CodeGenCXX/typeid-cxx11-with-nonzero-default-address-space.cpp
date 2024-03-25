// RUN: %clang_cc1 -I%S %s -triple spirv64-unknown-unknown -fsycl -fsycl-is-device -emit-llvm -std=c++11 -o - | FileCheck %s
#include "typeinfo"

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

extern constexpr Item items[] = {
  item<A>("A"), item<B>("B"), item<C>("C"), item<int>("int")
};

constexpr auto &x = items[0].ti;

constexpr auto &y = typeid(B{});

}
