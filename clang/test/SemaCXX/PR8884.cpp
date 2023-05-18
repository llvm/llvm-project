// RUN: %clang_cc1 -fsyntax-only  %s
extern "C" {
  class bar {
    friend struct foo;
    static struct foo& baz ();
  };
  struct foo {
    void zed () {
      bar::baz();
    }
  };
}
