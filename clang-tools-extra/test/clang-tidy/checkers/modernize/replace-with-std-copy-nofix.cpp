// RUN: %check_clang_tidy %s modernize-replace-with-std-copy %t -- \
// RUN:   -config='{CheckOptions: {modernize-replace-with-std-copy.FlagMemcpy: true}}' \
// RUN:   | FileCheck %s -implicit-check-not="{{FIX-IT}}"

namespace std {
using size_t = decltype(sizeof(int));

void *memcpy(void *__restrict dest, const void *__restrict src, size_t n);
void *memmove(void *__restrict dest, const void *__restrict src, size_t n);

template <typename T> struct vector {
  vector(size_t);

  T *data();
  const T *data() const;
  size_t size() const;
  using value_type = T;
};

template<typename Container>
typename Container::value_type* data(Container& Arg)  {
    return Arg.data();
}

template<typename T>
T* data(T Arg[]);

template<typename Container>
const typename Container::value_type* data(const Container& Arg) {
    return Arg.data();
}

template<typename T>
const T* data(const T[]);

} // namespace std

namespace {
using size_t = std::size_t;

void *memmove(void *__restrict dest, const void *__restrict src, size_t n) {
  return nullptr;
}
} // namespace

namespace {
void noFixItEx() {
  { // different value type widths
    int Source[10];
    std::vector<char> Dest(5);

    memmove(Source, std::data(Dest), 1);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
    std::memmove(Source, std::data(Dest), 1);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
    memmove(std::data(Source), Dest.data(), 1);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
    std::memmove(std::data(Source), Dest.data(), 1);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
    memmove(std::data(Source), std::data(Dest), 1);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
    std::memmove(std::data(Source), std::data(Dest), 1);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
  }

  { // return value used
    int Source[10];
    std::vector<char> Dest(5);

    void* Vptr = nullptr;
    if (memmove(Source, Dest.data(), 1) != nullptr) {
      // CHECK-MESSAGES: [[@LINE-1]]:9: warning: prefer std::copy_n to 'memmove'
      Vptr = memmove(Source, Dest.data(), 1);
      // CHECK-MESSAGES: [[@LINE-1]]:14: warning: prefer std::copy_n to 'memmove'
    }
    Vptr = [&]() {
      return memmove(Source, Dest.data(), 1);
      // CHECK-MESSAGES: [[@LINE-1]]:14: warning: prefer std::copy_n to 'memmove'
    }();
    
    for (;Vptr != nullptr; Vptr = memmove(Source, Dest.data(), 1)) {}
    // CHECK-MESSAGES: [[@LINE-1]]:35: warning: prefer std::copy_n to 'memmove'
  }
}
} // namespace