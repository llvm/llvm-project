// RUN: %check_clang_tidy %s readability-use-builtin-literals %t

void warn_and_fix() {

  (char16_t)U'a';
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: u'a';
  (char32_t)u'a';
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: U'a';

  (int)1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 1;
  (unsigned int)0x1ul;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 0x1u;
  (long int)2l;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 2L;
  (unsigned long int)0x2lu;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 0x2uL;
  (long long int)3ll;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 3LL;
  (unsigned long long int)0x3llu;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 0x3uLL;

  (double)1.f;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 1.;
  (float)2.;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 2.f;
  (long double)3e0f;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 3e0L;

  float(2.);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 2.f;
  double{2.};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 2.;

  static_cast<double>(2.f);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 2.;

  reinterpret_cast<int>(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 1;
}

#define OPSHIFT ((unsigned)27)
#define OCHAR (2LU<<OPSHIFT)
#define OPSHIFT2 (27)
#define OCHAR2 (2LU<<(unsigned)OPSHIFT2)
#define SUBT unsigned
#define OCHAR3 (2LU<<(SUBT)27)
#define SUBT2 char
#define OCHAR4 (2LU<<(SUBT2)'a')

void warn_and_recommend_fix() {

  OPSHIFT;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in 'u' suffix instead of explicit cast to 'unsigned int' [readability-use-builtin-literals]
  OCHAR;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in 'u' suffix instead of explicit cast to 'unsigned int' [readability-use-builtin-literals]
  OCHAR2;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in 'u' suffix instead of explicit cast to 'unsigned int' [readability-use-builtin-literals]
  OCHAR3;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in 'u' suffix instead of explicit cast to 'unsigned int' [readability-use-builtin-literals]
  OCHAR4;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in '' prefix instead of explicit cast to 'char' [readability-use-builtin-literals]
}

#define INT_MAX 2147483647
#define MAXCOL 2

#ifdef CHECKED
#define SUINT int
#else
#define SUINT unsigned
#endif

template <typename T>
T f() {
  return T(1);
}

int no_warn() {

(void)0;
(unsigned*)0;

static_cast<unsigned>(INT_MAX);
(unsigned)MAXCOL;

(SUINT)31;

return f<int>();
}
