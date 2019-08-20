namespace header {
  class Z {
  public:
    Z() {
      foo(); // impure-warning {{Call to virtual method 'Z::foo' during construction bypasses virtual dispatch}}
    }
    virtual int foo();
  };
}
