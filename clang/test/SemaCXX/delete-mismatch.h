// Header for PCH test delete.cpp
namespace pch_test {
struct X {
  int *a;
  X();
  X(int);
  X(bool)
    : a(new int[1]) { } // expected-note{{allocated with 'new[]' here}}
  ~X()
  {
    delete a; // expected-warning{{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:9}:"[]"
  }
};
}
