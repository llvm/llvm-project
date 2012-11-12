// RUN: %clang_cc1 -fsyntax-only %s

template<typename T, T I, int J>
struct adder {
  enum {
    value = I + J,
    value2
  };
};

int array1[adder<long, 3, 4>::value == 7? 1 : -1];

namespace PR6375 {
  template<typename T> 
  void f() {
    enum Enum
    {
      enumerator1 = 0xFFFFFFF,
      enumerator2 = enumerator1 - 1
    };
  
    int xb1 = enumerator1;
    int xe1 = enumerator2;
  }

  template void f<int>();
}
