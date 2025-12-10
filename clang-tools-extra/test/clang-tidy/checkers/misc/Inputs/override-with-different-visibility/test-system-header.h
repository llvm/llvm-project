#pragma clang system_header

namespace sys {

struct Base {
  virtual void publicF();
};

struct Derived: public Base {
private:
  void publicF() override;
};

}
