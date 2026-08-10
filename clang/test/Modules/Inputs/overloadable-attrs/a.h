namespace enable_if_attrs {
constexpr int fn1() __attribute__((enable_if(0, ""))) { return 0; }
constexpr int fn1() { return 1; }

constexpr int fn2() { return 1; }
constexpr int fn2() __attribute__((enable_if(0, ""))) { return 0; }

constexpr int fn3(int i) __attribute__((enable_if(!i, ""))) { return 0; }
constexpr int fn3(int i) __attribute__((enable_if(i, ""))) { return 1; }

constexpr int fn4(int i) { return 0; }
constexpr int fn4(int i) __attribute__((enable_if(i, ""))) { return 1; }

constexpr int fn5(int i) __attribute__((enable_if(i, ""))) { return 1; }
constexpr int fn5(int i) { return 0; }
}

namespace pass_object_size_attrs {
constexpr int fn1(void *const a __attribute__((pass_object_size(0)))) {
  return 1;
}
constexpr int fn1(void *const a) { return 0; }

constexpr int fn2(void *const a) { return 0; }
constexpr int fn2(void *const a __attribute__((pass_object_size(0)))) {
  return 1;
}
}
