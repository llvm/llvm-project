// Like the compiler, the static analyzer treats some functions differently if
// they come from a system header -- for example, it is assumed that system
// functions do not arbitrarily free() their parameters, and that some bugs
// found in system headers cannot be fixed by the user and should be
// suppressed.

#pragma clang system_header

struct RecordInSystemHeader {
  int a;
  int b;
};

template <class T>
struct ContainerInSystemHeader {
  T &t;
  ContainerInSystemHeader(T& t) : t(t) {}
};
