namespace ns { template <typename T> class C; };
class A {
  template <typename T> friend class ::ns::C;
};
