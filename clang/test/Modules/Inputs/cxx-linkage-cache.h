// Reduced from a crash encountered with a modularized libc++, where
// we would try to compute the linkage of a declaration before we
// finish loading the relevant pieces of it.
inline namespace D {
  template<class>
  struct U {
    friend bool f(const U &);
  };

  template class U<int>;
}
