// RUN: %check_clang_tidy %s bugprone-suspicious-memory-comparison %t \
// RUN: -- -- -target i386-unknown-unknown

static_assert(sizeof(int *) == sizeof(int));

namespace std {
typedef __SIZE_TYPE__ size_t;
int memcmp(const void *lhs, const void *rhs, size_t count);
} // namespace std

namespace no_padding_on_32bit {
struct S {
  int x;
  int *y;
};

void test() {
  S a, b;
  std::memcmp(&a, &b, sizeof(S));
}
} // namespace no_padding_on_32bit

namespace inner_padding {
struct S {
  char x;
  int y;
};
void test() {
  S a, b;
  std::memcmp(&a, &b, sizeof(S));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing object representation of type 'inner_padding::S' which does not have a unique object representation; consider comparing the members of the object manually
}
} // namespace inner_padding
