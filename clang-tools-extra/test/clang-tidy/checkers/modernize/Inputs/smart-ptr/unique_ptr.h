namespace std {

template <typename T>
class default_delete {};

template <typename type, typename Deleter = std::default_delete<type>>
class unique_ptr {
public:
  unique_ptr() {}
  unique_ptr(type *ptr) {}
  unique_ptr(const unique_ptr<type> &t) = delete;
  unique_ptr(unique_ptr<type> &&t) {}
  ~unique_ptr() {}
  type &operator*() { return *ptr; }
  type *operator->() { return ptr; }
  type *release() { return ptr; }
  void reset() {}
  void reset(type *pt) {}
  void reset(type pt) {}
  unique_ptr &operator=(unique_ptr &&) { return *this; }
  template <typename T>
  unique_ptr &operator=(unique_ptr<T> &&) { return *this; }

private:
  type *ptr;
};

}  // namespace std
