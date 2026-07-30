// RUN: not %clang_cc1 -fsyntax-only %s
// Note: The important part here is that we don't crash, not any specific errors
class Test {
 public:
  Test() : ab_(false {};

  bool ab() {
    return ab_;
  }
 private:
  bool ab_;
}
