// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.MemoryUnsafeCastChecker -verify %s

// Types spelled before and after a redeclaration refer to different
// redeclarations of the same class. Casts between them must still be
// recognized as same-type, upcast, or downcast casts, not "unrelated" casts.

class RedeclBase {
public:
  virtual ~RedeclBase();
};

class RedeclDerived : public RedeclBase {
public:
  void method();
  void constMethod() const;
};

const RedeclDerived *getConstDerived();
RedeclDerived *getDerived();
RedeclBase *getBase();

class RedeclBase;
class RedeclDerived;

void RedeclDerived::constMethod() const {
  const_cast<RedeclDerived *>(this)->method();
}

void test_const_cast_redecl(const RedeclDerived &ref) {
  const_cast<RedeclDerived *>(getConstDerived())->method();
  const_cast<RedeclDerived &>(*getConstDerived()).method();
  const_cast<RedeclDerived &>(ref).method();
}

void test_upcast_redecl() {
  (void)static_cast<RedeclBase *>(getDerived());
  (void)static_cast<RedeclBase &>(*getDerived());
}

void test_downcast_redecl() {
  (void)static_cast<RedeclDerived *>(getBase());
  // expected-warning@-1{{Unsafe cast from base type 'RedeclBase' to derived type 'RedeclDerived'}}
  (void)static_cast<RedeclDerived &>(*getBase());
  // expected-warning@-1{{Unsafe cast from base type 'RedeclBase' to derived type 'RedeclDerived'}}
}
