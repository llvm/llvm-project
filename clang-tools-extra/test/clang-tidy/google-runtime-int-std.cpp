// RUN: %check_clang_tidy %s google-runtime-int %t -- \
// RUN:   -config='{CheckOptions: [ \
// RUN:     {key: google-runtime-int.UnsignedTypePrefix, value: "std::uint"}, \
// RUN:     {key: google-runtime-int.SignedTypePrefix, value: "std::int"}, \
// RUN:     {key: google-runtime-int.TypeSuffix, value: "_t"}, \
// RUN:   ]}'

long a();
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: consider replacing 'long' with 'std::int{{..}}_t'

typedef unsigned long long uint64; // NOLINT

long b(long = 1);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: consider replacing 'long' with 'std::int{{..}}_t'
// CHECK-MESSAGES: [[@LINE-2]]:8: warning: consider replacing 'long' with 'std::int{{..}}_t'

template <typename T>
void tmpl() {
  T i;
}

short bar(const short, unsigned short) {
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: consider replacing 'short' with 'std::int16_t'
// CHECK-MESSAGES: [[@LINE-2]]:17: warning: consider replacing 'short' with 'std::int16_t'
// CHECK-MESSAGES: [[@LINE-3]]:24: warning: consider replacing 'unsigned short' with 'std::uint16_t'
  long double foo = 42;
  uint64 qux = 42;
  unsigned short port;

  const unsigned short bar = 0;
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: consider replacing 'unsigned short' with 'std::uint16_t'
  long long *baar;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'long long' with 'std::int64_t'
  const unsigned short &bara = bar;
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: consider replacing 'unsigned short' with 'std::uint16_t'
  long const long moo = 1;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'long long' with 'std::int64_t'
  long volatile long wat = 42;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'long long' with 'std::int64_t'
  unsigned long y;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'unsigned long' with 'std::uint{{..}}_t'
  unsigned long long **const *tmp;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'unsigned long long' with 'std::uint64_t'
  unsigned long long **const *&z = tmp;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'unsigned long long' with 'std::uint64_t'
  unsigned short porthole;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'unsigned short' with 'std::uint16_t'

  uint64 cast = (short)42;
// CHECK-MESSAGES: [[@LINE-1]]:18: warning: consider replacing 'short' with 'std::int16_t'

#define l long
  l x;

  tmpl<short>();
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: consider replacing 'short' with 'std::int16_t'
}
