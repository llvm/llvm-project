// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

template <typename T>
class RefPtr {
public:
  RefPtr(T* ptr)
    : m_ptr(ptr)
  {
    if (m_ptr)
      m_ptr->ref();
  }

  ~RefPtr()
  {
    if (m_ptr)
      m_ptr->deref();
  }

  T* get() { return m_ptr; }

private:
  T* m_ptr;
};

class Obj {
public:
  static Obj* get();
  static RefPtr<Obj> create();
  void ref() const;
  void deref() const;
};

void someFunction(Obj*, Obj* = nullptr);
void otherFunction(Obj*, Obj* = Obj::get());
// expected-warning@-1{{Call argument is uncounted and unsafe [alpha.webkit.UncountedCallArgsChecker]}}
void anotherFunction(Obj*, Obj* = Obj::create().get());

void otherFunction() {
  someFunction(nullptr);
  someFunction(Obj::get());
  // expected-warning@-1{{Call argument is uncounted and unsafe [alpha.webkit.UncountedCallArgsChecker]}}
  someFunction(Obj::create().get());
  otherFunction(nullptr);
  anotherFunction(nullptr);
}
