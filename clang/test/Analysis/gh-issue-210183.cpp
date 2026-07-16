// RUN: %clang_analyze_cc1 -analyzer-checker=core -std=c++17 -verify %s

// https://github.com/llvm/llvm-project/issues/210183
// Don't crash when binding a value that isn't a CompoundVal (e.g. a pointer
// reinterpreted as an integer via a template-induced round trip) to a region
// of array type.

// expected-no-diagnostics

template <typename T>
class Span {
public:
  template <int N>
  Span(T (&arr)[N]) : ptr_(arr) {}
  T *data() { return ptr_; }
  T *ptr_;
};

class Decoder {
public:
  template <typename TInput, typename TOutput>
  auto decodeOne(TInput input, TOutput output) {
    using InPtr_t = decltype(input.data());
    InPtr_t inPtr = input.data();
    using OutPtr_t = decltype(output.data());
    OutPtr_t outPtr = output.data();
    int scratch[1];
    scratch[0] = *inPtr;
    int value = scratch[0];
    *outPtr = value;
  }
};

void test() {
  char buffer[1];
  Span<char> span(buffer);
  Decoder decoder;
  decoder.decodeOne(span, span);
}
