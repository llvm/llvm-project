// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.security.taint.TaintPropagation \
// RUN:   -analyzer-checker=debug.ExprInspection

// See issue https://github.com/llvm/llvm-project/issues/62663

template <typename T> void clang_analyzer_dump(T);
void clang_analyzer_warnIfReached();
void clang_analyzer_numTimesReached();
void clang_analyzer_isTainted(int);

extern int scanf(const char *format, ...);

class ActionHandler {
public:
  virtual ~ActionHandler() = default;
  virtual void onAction(int x, int &) {
    clang_analyzer_dump(x + 1); // expected-warning {{101}}
  }
};

class MyHandler final : public ActionHandler {
public:
  void onAction(int x, int &) override {
    clang_analyzer_dump(x + 2); // expected-warning {{202}}
  }
};

class MyOtherHandler final : public ActionHandler {
public:
  void onAction(int x, int &) override {
    clang_analyzer_dump(x + 3); // expected-warning {{403}}
  }
};

void trust_static_types(ActionHandler *p) {
  // This variable will help to see if conservative call evaluation happened or not.
  int invalidation_detector;

  // At this point we don't know anything about the dynamic type of `*p`, thus the `onAction` call might be resolved to the default implementation, matching the static type.
  invalidation_detector = 1000;
  p->onAction(100, invalidation_detector);
  clang_analyzer_dump(invalidation_detector);
  // expected-warning@-1 {{1000}} on this path we trusted the type, and resolved the call to `ActionHandler::onAction(int, int&)`
  // expected-warning@-2 {{conj}} on this path we conservatively evaluated the previous call

  clang_analyzer_numTimesReached(); // expected-warning {{2}} we only have that 2 paths here
  // 1) inlined `ActionHandler::onAction(int, int&)`
  // 2) conservatively eval called

  // Trust that the `*p` refers to an object with `MyHandler` static type (or to some other sub-class).
  auto *q = static_cast<MyHandler *>(p);
  (void)q; // Discard the result of the cast. We already learned the type `p` might refer to.

  invalidation_detector = 2000;
  p->onAction(200, invalidation_detector);
  clang_analyzer_dump(invalidation_detector);
  // expected-warning@-1 {{2000}} on this path we trusted the type, and resolved the call to `MyHandler::onAction(int, int&)`
  // expected-warning@-2 {{conj}} on this path we conservatively evaluated the previous call

  clang_analyzer_numTimesReached(); // expected-warning {{3}} we only have 3 paths here, not 4
  // 1) inlined 2 different callees
  // 2) inlined only 1st
  // 3) none were inlined
  // 4) inlined only the second: This can't happen because if we conservative called a specific function on a path, we will always evaluate it like that.
  //    See ExprEngine::BifurcateCall and DynamicDispatchBifurcationMap.
}


void conflicting_casts(ActionHandler *p) {
  (void)static_cast<MyHandler *>(p);
  (void)static_cast<MyOtherHandler *>(p);
  int invalidation_detector = 4000;
  p->onAction(400, invalidation_detector);
  clang_analyzer_dump(invalidation_detector);
  // expected-warning@-1 {{4000}}
  // expected-warning@-2 {{conj}}
}

// -------

class Base {
public:
  virtual void OnRecvCancel(int port) = 0;
};
class Handler final : public Base {
public:
  void OnRecvCancel(int port) override {
    clang_analyzer_dump(100 + port); // expected-warning {{+ 150}}
    clang_analyzer_isTainted(port);  // expected-warning {{YES}}
  }
};
class PParent  {
public:
  bool OnMessageReceived();
};
class Actor {
public:
  explicit Actor(Base* aRequest) : m(aRequest) {}
protected:
  Base* m;
};

class Parent : public Actor, public PParent {
public:
  explicit Parent(Base* aRequest) : Actor(aRequest) {}
  bool RecvCancel(int port) {
    clang_analyzer_dump(200 + port); // expected-warning {{+ 200}}
    clang_analyzer_isTainted(port);  // expected-warning {{YES}}
    Handler* foo = (Handler*)m;
    foo->OnRecvCancel(port + 50);
    return true;
  }
};

bool PParent::OnMessageReceived() {
  int port;
  scanf("%i", &port);
  clang_analyzer_isTainted(port); // expected-warning {{YES}}
  Parent* foo = static_cast<Parent*>(this);
  return foo->RecvCancel(port);
}
