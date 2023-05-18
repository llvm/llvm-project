
// NOTE: source_location.cpp must include this file after defining
// std::source_location.
namespace source_location_file {

constexpr const char *FILE = __FILE__;
constexpr const char *FILE_NAME = __FILE_NAME__;

constexpr SL global_info = SL::current();
constexpr const char *global_info_filename = __builtin_FILE_NAME();

constexpr SL test_function(SL v = SL::current()) {
  return v;
}

constexpr SL test_function_indirect() {
  return test_function();
}

constexpr const char *test_function_filename(
                      const char *file_name = __builtin_FILE_NAME()) {
  return file_name;
}

constexpr const char *test_function_filename_indirect() {
  return test_function_filename();
}

template <class T, class U = SL>
constexpr U test_function_template(T, U u = U::current()) {
  return u;
}

template <class T, class U = SL>
constexpr U test_function_template_indirect(T t) {
  return test_function_template(t);
}

template <class T, class U = const char *>
constexpr U test_function_filename_template(T, U u = __builtin_FILE_NAME()) {
  return u;
}

template <class T, class U = const char *>
constexpr U test_function_filename_template_indirect(T t) {
  return test_function_filename_template(t);
}

struct TestClass {
  SL info = SL::current();
  const char *info_file_name = __builtin_FILE_NAME();
  SL ctor_info;
  const char *ctor_info_file_name = nullptr;
  TestClass() = default;
  constexpr TestClass(int, SL cinfo = SL::current(),
                      const char *cfile_name = __builtin_FILE_NAME()) :
                      ctor_info(cinfo), ctor_info_file_name(cfile_name) {}
  template <class T, class U = SL>
  constexpr TestClass(int, T, U u = U::current(),
                      const char *cfile_name = __builtin_FILE_NAME()) :
                      ctor_info(u), ctor_info_file_name(cfile_name) {}
};

template <class T = SL>
struct AggrClass {
  int x;
  T info;
  T init_info = T::current();
  const char *init_info_file_name = __builtin_FILE_NAME();
};

} // namespace source_location_file
