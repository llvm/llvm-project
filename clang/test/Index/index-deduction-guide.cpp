// RUN: c-index-test core -print-source-symbols -- %s -std=gnu++17 | FileCheck %s

template<typename T>
typename T::type declval() {}
template <typename T> struct Test;
template <typename C, typename T = decltype(declval<C>().d())> Test(C &) -> Test<T>;
// CHECK: [[@LINE-1]]:45 | function/C | declval
// CHECK-NOT: RelCall
// CHECK: [[@LINE-3]]:77 | struct(Gen)/C++ | Test
// CHECK: [[@LINE-4]]:64 | struct(Gen)/C++ | Test
