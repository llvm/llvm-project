// RUN: %clang_cc1 %s -emit-llvm-only
// PR7040
struct fake_tuple;
struct connection {
    void bar(fake_tuple);
};
void (connection::*a)(fake_tuple) = &connection::bar;
void f() {
  void (connection::*b)(fake_tuple) = &connection::bar;
}
