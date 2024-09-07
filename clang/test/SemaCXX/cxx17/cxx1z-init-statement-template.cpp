// RUN: %clang_cc1 -std=c++1z -verify -emit-llvm-only %s
// expected-no-diagnostics

template <unsigned int BUFFER_SIZE> class Buffer {};

class A {
public:
  int status;
};

template <unsigned int N> A parse(Buffer<N> buffer);

template<unsigned int N>
void init_in_if(Buffer<N> buffer) {
  if (A a = parse(buffer); a.status > 0) {
  }
}

template<unsigned int N>
void init_in_switch(Buffer<N> buffer) {
  switch (A a = parse(buffer); a.status) {
    default:
      break;
  }
}

void test() {
  Buffer<10> buffer;
  init_in_if(buffer);
  init_in_switch(buffer);
}
