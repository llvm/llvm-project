// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct S {
  __attribute__((flatten_depth(T::value)))
  void func() {}
};

struct HasFloat {
  static constexpr float value = 3.14f;
};

struct HasPointer {
  static constexpr int* value = nullptr;
};

struct HasString {
  static constexpr const char* value = "bad";
};

void test() {
  S<HasFloat> s1; // expected-note {{in instantiation of template class 'S<HasFloat>' requested here}}
  S<HasPointer> s2; // expected-note {{in instantiation of template class 'S<HasPointer>' requested here}}
  S<HasString> s3; // expected-note {{in instantiation of template class 'S<HasString>' requested here}}
}

// expected-error@5 {{'flatten_depth' attribute requires an integer constant}}
// expected-error@5 {{'flatten_depth' attribute requires an integer constant}}
// expected-error@5 {{'flatten_depth' attribute requires an integer constant}}
