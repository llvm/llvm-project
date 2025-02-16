// This is a fake system header with divide-by-zero bugs introduced in
// c++ std library functions. We use these bugs to test hard-coded
// suppression of diagnostics within standard library functions that are known
// to produce false positives.

#pragma clang system_header
namespace std {
struct mutex {
  void lock() {}
  void unlock() {}
};

template <typename T> struct lock_guard {
  lock_guard<T>(std::mutex) {}
  ~lock_guard<T>() {}
};
} // namespace std
