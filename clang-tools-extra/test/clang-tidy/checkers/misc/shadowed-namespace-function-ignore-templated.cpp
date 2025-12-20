// RUN: %check_clang_tidy -std=c++98-or-later %s misc-shadowed-namespace-function %t -- \
// RUN:   --config="{CheckOptions: {misc-shadowed-namespace-function.IgnoreTemplated: true}}"

// Test that template instantiations are checked by default
namespace foo {
  template<typename T>
  void f_template();
}

// Function template definition - this is NOT a template instantiation, so it should always warn
template<typename T>
void f_template() {}

// Explicit template instantiation - this IS a template instantiation
template void f_template<int>();
// When IgnoreTemplated is true, the warning should NOT appear

// Test with another template
namespace bar {
  template<typename T>
  void g_template();
}

template<typename T>
void g_template() {}

const int _ = (g_template<char>(), 0);

// Test with another template
namespace bar2 {
  template<typename T>
  void j_template();
}

template<typename T>
void j_template() {}
