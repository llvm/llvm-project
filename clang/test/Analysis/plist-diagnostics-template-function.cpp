// RUN: %clang_analyze_cc1 -analyzer-output=plist -o %t.plist -std=c++11 -analyzer-checker=core %s
// RUN: FileCheck --input-file=%t.plist %s

bool ret();

template <class T>
void f(int i) {
  if (ret())
    i = i / (i - 5);
}

template <>
void f<int>(int i) {
  if (ret())
    i = i / (i - 5);
}

template <int N = 0>
void defaultTemplateParameterFunction(int i) {
  if (ret())
    int a = 10 / i;
}

template <typename... Args>
void variadicTemplateFunction(int i) {
  if (ret())
    int a = 10 / i;
}

int main() {
  f<int>(5);
  f<float>(5);
  defaultTemplateParameterFunction<>(0);
  variadicTemplateFunction<char, float, double, int *>(0);
}

// CHECK:      <string>Calling &apos;f&lt;float&gt;&apos;</string>
// CHECK:      <string>Calling &apos;f&lt;int&gt;&apos;</string>
// CHECK:      <string>Calling &apos;defaultTemplateParameterFunction&lt;0&gt;&apos;</string>
// CHECK:      <string>Calling &apos;variadicTemplateFunction&lt;char, float, double, int *&gt;&apos;</string>

