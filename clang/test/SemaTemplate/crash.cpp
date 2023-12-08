// RUN: not %clang_cc1 -verify %s -std=c++11

// PR17730
template <typename T>
void S<T>::mem1();

template <typename T>
void S<T>::mem2() {
    const int I = sizeof(T);
      (void)I;
}
