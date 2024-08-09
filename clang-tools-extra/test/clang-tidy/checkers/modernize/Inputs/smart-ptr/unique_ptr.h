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
  type *get() const;
  type& operator[](unsigned i) const;
  void reset() {}
  void reset(type *pt) {}
  void reset(type pt) {}
  explicit operator bool() const;
  unique_ptr &operator=(unique_ptr &&) { return *this; }
  template <typename T>
  unique_ptr &operator=(unique_ptr<T> &&) { return *this; }

private:
  type *ptr;
};

template <typename>
struct remove_reference;

template <typename _Tp>
struct remove_reference {
  typedef _Tp type;
};

template <typename _Tp>
struct remove_reference<_Tp &> {
  typedef _Tp type;
};

template <typename _Tp>
struct remove_reference<_Tp &&> {
  typedef _Tp type;
};

template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&move(_Tp &&__t) noexcept {
  return static_cast<typename remove_reference<_Tp>::type &&>(__t);
}

template <class _Tp>
constexpr _Tp&&
forward(typename std::remove_reference<_Tp>::type& __t) noexcept {
  return static_cast<_Tp&&>(__t);
}

template <class _Tp>
constexpr _Tp&&
forward(typename std::remove_reference<_Tp>::type&& __t) noexcept {
  return static_cast<_Tp&&>(__t);
}

template <typename T, typename... Args> std::unique_ptr<T> make_unique(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace std
