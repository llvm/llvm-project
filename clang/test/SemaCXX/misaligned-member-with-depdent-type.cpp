// RUN: %clang_cc1 -fsyntax-only -verify %s

struct __attribute__((packed)) {
  unsigned options;
  template <typename T>
  void getOptions() {
      (T *)&options;
  }
  template <typename U>
	void getOptions2() {
      (U)&options;
	}
} s;

struct __attribute__((packed)) { // expected-error {{anonymous structs and classes must be class members}}
           unsigned options ;
  template <typename T> getOptions() // expected-error {{a type specifier is required for all declarations}}
    {
      (T *) & options;
    }
};
