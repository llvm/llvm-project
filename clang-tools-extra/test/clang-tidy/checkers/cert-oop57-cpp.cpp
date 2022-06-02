// RUN: %check_clang_tidy %s cert-oop57-cpp %t -- \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: cert-oop57-cpp.MemSetNames, value: mymemset}, \
// RUN:  {key: cert-oop57-cpp.MemCpyNames, value: mymemcpy}, \
// RUN:  {key: cert-oop57-cpp.MemCmpNames, value: mymemcmp}]}' \
// RUN: --

void mymemset(void *, unsigned char, decltype(sizeof(int)));
void mymemcpy(void *, const void *, decltype(sizeof(int)));
int mymemcmp(const void *, const void *, decltype(sizeof(int)));

namespace std {
void memset(void *, unsigned char, decltype(sizeof(int)));
void memcpy(void *, const void *, decltype(sizeof(int)));
void memmove(void *, const void *, decltype(sizeof(int)));
void strcpy(void *, const void *, decltype(sizeof(int)));
int memcmp(const void *, const void *, decltype(sizeof(int)));
int strcmp(const void *, const void *, decltype(sizeof(int)));
} // namespace std

struct Trivial {
  int I;
  int J;
};

struct NonTrivial {
  int I;
  int J;

  NonTrivial() : I(0), J(0) {}
  NonTrivial &operator=(const NonTrivial &Other) {
    I = Other.I;
    J = Other.J;
    return *this;
  }

  bool operator==(const Trivial &Other) const {
    return I == Other.I && J == Other.J;
  }
  bool operator!=(const Trivial &Other) const {
    return !(*this == Other);
  }
};

void foo(const Trivial &Other) {
  Trivial Data;
  std::memset(&Data, 0, sizeof(Data));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: calling 'memset' on a non-trivially default constructible class is undefined
  std::memset(&Data, 0, sizeof(Trivial));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: calling 'memset' on a non-trivially default constructible class is undefined
  std::memcpy(&Data, &Other, sizeof(Data));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: calling 'memcpy' on a non-trivially copyable class is undefined
  std::memmove(&Data, &Other, sizeof(Data));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: calling 'memmove' on a non-trivially copyable class is undefined
  std::strcpy(&Data, &Other, sizeof(Data));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: calling 'strcpy' on a non-trivially copyable class is undefined
  std::memcmp(&Data, &Other, sizeof(Data));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: consider using comparison operators instead of calling 'memcmp'
  std::strcmp(&Data, &Other, sizeof(Data));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: consider using comparison operators instead of calling 'strcmp'
}

void bar(const NonTrivial &Other) {
  NonTrivial Data;
  std::memset(&Data, 0, sizeof(Data));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling 'memset' on a non-trivially default constructible class is undefined
  // Check it detects sizeof(Type) as well as sizeof(Instantiation)
  std::memset(&Data, 0, sizeof(NonTrivial));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling 'memset' on a non-trivially default constructible class is undefined
  std::memcpy(&Data, &Other, sizeof(Data));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling 'memcpy' on a non-trivially copyable class is undefined
  std::memmove(&Data, &Other, sizeof(Data));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling 'memmove' on a non-trivially copyable class is undefined
  std::strcpy(&Data, &Other, sizeof(Data));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling 'strcpy' on a non-trivially copyable class is undefined
  std::memcmp(&Data, &Other, sizeof(Data));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: consider using comparison operators instead of calling 'memcmp'
  std::strcmp(&Data, &Other, sizeof(Data));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: consider using comparison operators instead of calling 'strcmp'
}

void baz(const NonTrivial &Other) {
  NonTrivial Data;
  mymemset(&Data, 0, sizeof(Data));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling 'mymemset' on a non-trivially default constructible class is undefined
  mymemcpy(&Data, &Other, sizeof(Data));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling 'mymemcpy' on a non-trivially copyable class is undefined
  mymemcmp(&Data, &Other, sizeof(Data));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: consider using comparison operators instead of calling 'mymemcmp'
}
