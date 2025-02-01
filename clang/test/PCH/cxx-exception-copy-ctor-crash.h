// Header for PCH test cxx-exception-copy-ctor-crash.cpp

inline int ctor_count = 0;
inline int dtor_count = 0;

struct Exception {
  Exception() { ++ctor_count; }
  ~Exception() { ++dtor_count; }
  Exception(const Exception &) noexcept { ++ctor_count; }
};

inline void throw_exception() {
  throw Exception();
}
