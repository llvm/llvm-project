// RUN: %clang_cc1 -std=c++2a -triple %itanium_abi_triple -emit-llvm -o - %s -w | FileCheck %s

template<class, int, class>
struct DummyType { };

inline void inline_func() {
  // CHECK: UlvE
  []{}();

  // CHECK: UlTyvE
  []<class>{}.operator()<int>();

  // CHECK: UlTyT_E
  []<class T>(T){}(1);

  // CHECK: UlTyTyT_T0_E
  []<class T1, class T2>(T1, T2){}(1, 2);

  // CHECK: UlTyTyT0_T_E
  []<class T1, class T2>(T2, T1){}(2, 1);

  // CHECK: UlTniTyTnjT0_E
  []<int I, class T, unsigned U>(T){}.operator()<1, int, 2>(3);

  // CHECK: UlTyTtTyTniTyETniTyvE
  []<class,
     template<class, int, class> class,
     int,
     class>{}.operator()<unsigned, DummyType, 5, int>();
}

void call_inline_func() {
  inline_func();
}
