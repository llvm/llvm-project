// RUN: %clang -fplugin=%llvmshlibdir/CallSuperAttr%pluginext -fsyntax-only -Xclang -verify=callsuper %s
// RUN: %clang -fplugin=%llvmshlibdir/CallSuperAttr%pluginext -DBAD_CALLSUPER -fsyntax-only -Xclang -verify=badcallsuper %s
// REQUIRES: plugins, examples

// callsuper-no-diagnostics
struct Base1 {
  [[clang::call_super]] virtual void Test() {}
};
struct Base2 {
  [[clang::call_super]] virtual void Test() {}
};
struct Derive : public Base1, public Base2 {
#ifndef BAD_CALLSUPER
  void Test() override;
#else
  [[clang::call_super]] virtual void Test() override final;
  // badcallsuper-warning@16 {{'call_super' attribute marked on a final method}}
#endif
};
void Derive::Test() {
  Base1::Test();
#ifndef BAD_CALLSUPER
  Base2::Test();
#else
  // badcallsuper-warning@20 {{virtual function 'Base2::Test' is marked as 'call_super' but this overriding method does not call the base version}}
  // badcallsuper-note@10 {{function marked 'call_super' here}}
#endif
}
struct Derive2 : public Base1, public Base2 {
  void Test() override {
    Base1::Test();
    Base2::Test();
  }
};
