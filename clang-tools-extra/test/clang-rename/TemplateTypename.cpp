// Currently unsupported test.
// RUN: cat %s > %t.cpp
// FIXME: clang-rename should be able to rename template parameters correctly.

template <typename T>
T foo(T arg, T& ref, T* ptr) {
  T value;
  int number = 42;
  value = (T)number;
  value = static_cast<T>(number);
  return value;
}
