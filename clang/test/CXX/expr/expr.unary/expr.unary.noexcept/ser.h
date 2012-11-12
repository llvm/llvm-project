// Serialization testing helper for noexcept, included by cg.cpp.

inline bool f1() {
  return noexcept(0);
}
inline bool f2() {
  return noexcept(throw 0);
}
