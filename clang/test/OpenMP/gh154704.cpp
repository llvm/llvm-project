// RUN: %clang_cc1 -verify -fopenmp -fsyntax-only %s

template <typename T> struct Less {};

template <typename K, typename V, typename C = Less<K>> struct Map {
  V &operator[](const K &);
};

void no_crash() {
  int keys[42], data[42];
  Map<int, int> map;

#pragma omp target
  {
    for (int i = 0; i < 42; ++i)
      map[keys[i]] = data[i];
  }
}

template <typename T> struct Fails {
  typename T::type t; // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
};

template <typename T, typename U = Fails<T>> struct Holder {};

void point_of_instantiation() {
  Holder<int> h;

#pragma omp target
  {
    (void)&h; // expected-note {{in instantiation of template class 'Fails<int>' requested here}}
  }
}
