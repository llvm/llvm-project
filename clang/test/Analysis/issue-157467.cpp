// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// expected-no-diagnostics

template <class T, int Idx, bool CanBeEmptyBase = __is_empty(T) && (!__is_final(T))>
struct compressed_pair_elem {
  explicit compressed_pair_elem(T u) : value(u) {}
  T value;
};

template <class T, int Idx>
struct compressed_pair_elem<T, Idx, /*CanBeEmptyBase=*/true> : T {
  explicit compressed_pair_elem(T u) : T(u) {}
};

template <class T1, class T2, class Base1 = compressed_pair_elem<T1, 0>, class Base2 = compressed_pair_elem<T2, 1>>
struct compressed_pair : Base1, Base2 {
  explicit compressed_pair(T1 t1, T2 t2) : Base1(t1), Base2(t2) {}
};

// empty deleter object
template <class T>
struct default_delete {
  void operator()(T* p) {
    delete p;
  }
};

template <class T, class Deleter = default_delete<T> >
struct some_unique_ptr {
  // compressed_pair will employ the empty base class optimization, thus overlapping
  // the `int*` and the empty `Deleter` object, clobbering the pointer.
  compressed_pair<int*, Deleter> ptr;
  some_unique_ptr(int* p, Deleter d) : ptr(p, d) {}
  ~some_unique_ptr();
};

void entry_point() {
  some_unique_ptr<int, default_delete<int> > u3(new int(12), default_delete<int>());
}
