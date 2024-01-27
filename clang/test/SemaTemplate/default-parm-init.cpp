// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s
// expected-no-diagnostics

template<typename TemplateParam>
struct Problem{
  template<typename FunctionTemplateParam>
  constexpr int FuncAlign(int param = alignof(FunctionTemplateParam));

  template<typename FunctionTemplateParam>
  constexpr int FuncSizeof(int param = sizeof(FunctionTemplateParam));

  template<typename FunctionTemplateParam>
  constexpr int FuncAlign2(int param = alignof(TemplateParam));

  template<typename FunctionTemplateParam>
  constexpr int FuncSizeof2(int param = sizeof(TemplateParam));
};

template <>
template<typename FunctionTemplateParam>
constexpr int Problem<int>::FuncAlign(int param) {
	return param;
}

template <>
template<typename FunctionTemplateParam>
constexpr int Problem<int>::FuncSizeof(int param) {
	return param;
}

template <>
template<typename FunctionTemplateParam>
constexpr int Problem<int>::FuncAlign2(int param) {
	return param;
}

template <>
template<typename FunctionTemplateParam>
constexpr int Problem<int>::FuncSizeof2(int param) {
	return param;
}

int main(){
    Problem<int> p = {};
    static_assert(p.FuncAlign<char>() == alignof(char));
    static_assert(p.FuncSizeof<char>() == sizeof(char));
    static_assert(p.FuncAlign2<char>() == alignof(int));
    static_assert(p.FuncSizeof2<char>() == sizeof(int));
}
