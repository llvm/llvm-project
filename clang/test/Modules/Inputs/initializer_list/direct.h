namespace std {
  using size_t = decltype(sizeof(0));

  template<typename T> struct initializer_list {
    const T* ptr; size_t sz;
  };

  template<typename T> int min(initializer_list<T>);
}
