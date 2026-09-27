// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace InitPriorityAttribute {
  struct S1 {} s1;
  struct S2 {} s2; // #S2_INIT_PRIORITY
  [[gnu::init_priority(1000)]] auto auto_var = s1;
  [[gnu::init_priority(1000)]] S1 struct_var = s1;
  [[gnu::init_priority(1000)]] S2 invalid_var = s1; // expected-error {{no viable conversion from 'struct S1' to 'S2'}} \
                                                       expected-note@#S2_INIT_PRIORITY {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'struct S1' to 'const S2 &' for 1st argument}} \
                                                       expected-note@#S2_INIT_PRIORITY {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'struct S1' to 'S2 &&' for 1st argument}}

  template<typename T>
  struct ST { // #ST_INIT_PRIORITY
    T inner;
  };

  [[gnu::init_priority(2000)]] auto auto_t_var = ST<int>{0};
  [[gnu::init_priority(2000)]] ST<int> struct_t_var = ST<int>{0};
  [[gnu::init_priority(2000)]] ST<float> invalid_t_var = ST<int>{0}; // expected-error {{no viable conversion from 'ST<int>' to 'ST<float>'}} \
                                                                             expected-note@#ST_INIT_PRIORITY {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'ST<int>' to 'const ST<float> &' for 1st argument}} \
                                                                             expected-note@#ST_INIT_PRIORITY {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'ST<int>' to 'ST<float> &&' for 1st argument}}
}
