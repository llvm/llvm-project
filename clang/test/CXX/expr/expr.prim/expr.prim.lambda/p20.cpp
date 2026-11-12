// RUN: %clang_cc1 -std=c++11 %s -Wunused -verify
// expected-no-diagnostics

template<typename T>
void destroy(T* ptr) {
  ptr->~T();
  (*ptr).~T();
}

void destructor() {
  auto lambda = []{};
  destroy(&lambda);
}
