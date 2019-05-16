
// NOTE: source_location.cpp must include this file after defining
// std::source_location.
namespace source_location_file {

constexpr const char *FILE = __FILE__;

constexpr SL global_info = SL::current();

constexpr SL test_function(SL v = SL::current()) {
  return v;
}

constexpr SL test_function_indirect() {
  return test_function();
}

template <class T, class U = SL>
constexpr U test_function_template(T, U u = U::current()) {
  return u;
}

template <class T, class U = SL>
constexpr U test_function_template_indirect(T t) {
  return test_function_template(t);
}

struct TestClass {
  SL info = SL::current();
  SL ctor_info;
  TestClass() = default;
  constexpr TestClass(int, SL cinfo = SL::current()) : ctor_info(cinfo) {}
  template <class T, class U = SL>
  constexpr TestClass(int, T, U u = U::current()) : ctor_info(u) {}
};

template <class T = SL>
struct AggrClass {
  int x;
  T info;
  T init_info = T::current();
};

} // namespace source_location_file
