// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -Wunused

// PR9968: We used to warn that __range is unused in a dependent for-range.

template <typename T>
  struct Vector {
    void doIt() {
      int a; // expected-warning {{unused variable 'a'}}

      for (auto& e : elements)
        ;
    }

    T elements[10];
  };


int main(int, char**) {
  Vector<int>    vector;
  vector.doIt();
}
