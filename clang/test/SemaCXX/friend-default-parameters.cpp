// RUN: %clang_cc1 -std=c++20 -verify -emit-llvm-only %s

template <int>
void Create(const void* = nullptr);

template <int>
struct ObjImpl {
  template <int>
  friend void ::Create(const void*);
};

template <int I>
void Create(const void*) {
  (void) ObjImpl<I>{};
}

int main() {
  Create<42>();
}

// expected-no-diagnostics
