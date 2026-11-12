#pragma clang system_header

#ifndef mock_std_lifetime_h
#define mock_std_lifetime_h

using size_t = decltype(sizeof(0));
void *operator new(size_t, void *) noexcept;

namespace std {

template <typename T, typename... Args> T *construct_at(T *p, Args &...args) {
  return ::new (static_cast<void *>(p)) T(args...);
}

template <typename T> void destroy_at(T *p) { p->~T(); }

struct in_place_t {};
inline constexpr in_place_t in_place {};

template <typename T> class optional {
public:
  optional() {}
  template <typename... Args> optional(in_place_t, Args &...args) {
    emplace(args...);
  }
  ~optional() { reset(); }

  template <typename... Args> T &emplace(Args &...args) {
    reset();
    ::new (static_cast<void *>(&m_value)) T(args...);
    m_engaged = true;
    return m_value;
  }
  void reset() {
    if (m_engaged)
      m_value.~T();
    m_engaged = false;
  }
  T &operator*() { return m_value; }
  T *operator->() { return &m_value; }

private:
  union {
    char m_empty;
    T m_value;
  };
  bool m_engaged { false };
};

template <typename T> class unique_ptr {
public:
  explicit unique_ptr(T *p) : m_ptr(p) {}
  ~unique_ptr() { delete m_ptr; }
  T *get() const { return m_ptr; }

private:
  T *m_ptr;
};

template <typename R, typename...> struct coroutine_traits {
  using promise_type = typename R::promise_type;
};

template <typename P = void> struct coroutine_handle {
  static coroutine_handle from_address(void *) noexcept;
};

struct suspend_never {
  bool await_ready() noexcept { return true; }
  template <typename P> void await_suspend(coroutine_handle<P>) noexcept {}
  void await_resume() noexcept {}
};

} // namespace std

#endif
