// RUN: %check_clang_tidy %s google-runtime-int %t

long a();
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: consider replacing 'long' with 'int{{..}}'

typedef unsigned long long uint64; // NOLINT

long b(long = 1);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: consider replacing 'long' with 'int{{..}}'
// CHECK-MESSAGES: [[@LINE-2]]:8: warning: consider replacing 'long' with 'int{{..}}'

template <typename T>
void tmpl() {
  T i;
}

short bar(const short, unsigned short) {
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: consider replacing 'short' with 'int16'
// CHECK-MESSAGES: [[@LINE-2]]:17: warning: consider replacing 'short' with 'int16'
// CHECK-MESSAGES: [[@LINE-3]]:24: warning: consider replacing 'unsigned short' with 'uint16'
  long double foo = 42;
  uint64 qux = 42;
  unsigned short port;

  const unsigned short bar = 0;
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: consider replacing 'unsigned short' with 'uint16'
  long long *baar;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'long long' with 'int64'
  const unsigned short &bara = bar;
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: consider replacing 'unsigned short' with 'uint16'
  long const long moo = 1;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'long long' with 'int64'
  long volatile long wat = 42;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'long long' with 'int64'
  unsigned long y;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'unsigned long' with 'uint{{..}}'
  unsigned long long **const *tmp;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'unsigned long long' with 'uint64'
  unsigned long long **const *&z = tmp;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'unsigned long long' with 'uint64'
  unsigned short porthole;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'unsigned short' with 'uint16'

  uint64 cast = (short)42;
// CHECK-MESSAGES: [[@LINE-1]]:18: warning: consider replacing 'short' with 'int16'

#define l long
  l x;

  tmpl<short>();
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: consider replacing 'short' with 'int16'
  return 0;
}

void p(unsigned short port);

void qux() {
  short port;
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: consider replacing 'short' with 'int16'
}

// FIXME: This shouldn't warn, as UD-literal operators require one of a handful
// of types as an argument.
struct some_value {};
constexpr some_value operator"" _some_literal(unsigned long long int i);
// CHECK-MESSAGES: [[@LINE-1]]:47: warning: consider replacing 'unsigned long long'

struct A { A& operator=(const A&); };
class B { A a[0]; };

void fff() {
  B a, b;
  a = b;
}
