namespace std {
  using size_t = decltype(sizeof(0));

  template<typename T> struct initializer_list {
    initializer_list(T*, size_t);
  };

  template<typename T> int min(initializer_list<T>);
}
