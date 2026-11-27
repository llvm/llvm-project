class H {
  void operator delete(void *);
public:
  virtual ~H();
};
H::~H() { }

class S : public H {
  void operator delete(void *);
public:
  virtual ~S();
};
S::~S() { }

void in_h_tests() {
  H* h = new H();
  ::delete h;
}
