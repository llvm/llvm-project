// RUN: %clang_cc1 -std=c++11 -x c++-header %s -emit-pch -o %t.pch
// RUN: %clang_cc1 -std=c++11 -x c++ /dev/null -include-pch %t.pch

// RUN: %clang_cc1 -std=c++11 -x c++-header %s -emit-pch -fpch-instantiate-templates -o %t.pch
// RUN: %clang_cc1 -std=c++11 -x c++ /dev/null -include-pch %t.pch

class move_only { move_only(const move_only&) = delete; move_only(move_only&&); };
struct sb {
  move_only il;
  sb();
  sb(sb &&);
};

template<typename T> T make();
template<typename T> void doit(decltype(T(make<const T&>()))*) { T(make<const T&>()); }
template<typename T> void doit(...) { T(make<T&&>()); }
template<typename T> void later() { doit<T>(0); }

void fn1() {
  sb x;
  later<sb>();
}
