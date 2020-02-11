struct CommonBase {
  int m_base;
  int virt_base_val;
  int func_base_val;
  virtual int virt_base() { return virt_base_val; }
  virtual int virt_common() { return 555; }
  int func_base() { return func_base_val; }
  int func_common() { return 777; }
};

struct Base1 : CommonBase {
  int m1 = 22;
  Base1() {
    // Give the base functions/members unique values.
    virt_base_val = 111;
    func_base_val = 112;
    m_base = 11;
  }
  virtual int virt1() { return 3; }
  int func1() { return 4; }
};

struct Base2 : CommonBase {
  int m2 = 33;
  Base2() {
    // Give the base functions/members unique values.
    virt_base_val = 121;
    func_base_val = 122;
    m_base = 12;
  }
  virtual int virt2() { return 5; }
  int func2() { return 6; }
};

struct FinalClass : Base1, Base2 {
  int m_final = 44;
  virtual int final_virt() { return 7; }
  int final_func() { return 8; }
  virtual int virt_common() { return 444; }
  int func_common() { return 888; }
};

int main() {
  FinalClass C;
  // Call functions so they get emitted.
  C.func1();
  C.func2();
  C.final_func();
  C.func_common();
  C.Base1::func_base();
  return 0; // break here
}
