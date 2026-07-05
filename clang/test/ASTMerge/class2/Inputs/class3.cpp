class C1 {
public:
  C1();
  ~C1();
  C1 *method_1() {
    return this;
  }
  C1 method_2() {
    return C1();
  }
  void method_3() {
    const C1 &ref = C1();
  }
};

class C11 : public C1 {
};

class C2 {
private:
  int x;
  friend class C3;
public:
  static_assert(sizeof(x) == sizeof(int), "Error");
  typedef class C2::C2 InjType;
};
