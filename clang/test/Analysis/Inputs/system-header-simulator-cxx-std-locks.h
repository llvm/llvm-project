#pragma clang system_header

namespace std {
struct mutex {
  void lock();
  void unlock();
};

template <typename T> struct lock_guard {
  lock_guard(std::mutex &);
  ~lock_guard();
};
} // namespace std
