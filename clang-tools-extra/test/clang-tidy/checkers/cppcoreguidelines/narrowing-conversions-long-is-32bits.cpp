// RUN: %check_clang_tidy %s cppcoreguidelines-narrowing-conversions %t \
// RUN: -- -- -target x86_64-unknown-linux -m32

static_assert(sizeof(int) * 8 == 32, "int is 32-bits");
static_assert(sizeof(long) * 8 == 32, "long is 32-bits");
static_assert(sizeof(long long) * 8 == 64, "long long is 64-bits");

void narrow_integer_to_signed_integer_is_not_ok() {
  int i;        // i.e. int32_t
  long l;       // i.e. int32_t
  long long ll; // i.e. int64_t

  unsigned int ui;        // i.e. uint32_t
  unsigned long ul;       // i.e. uint32_t
  unsigned long long ull; // i.e. uint64_t

  i = l;  // int and long are the same type.
  i = ll; // int64_t does not fit in an int32_t
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'long long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  ll = ul;  // uint32_t fits into int64_t
  ll = ull; // uint64_t does not fit in an int64_t
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'unsigned long long' to signed type 'long long' is implementation-defined [cppcoreguidelines-narrowing-conversions]
}
