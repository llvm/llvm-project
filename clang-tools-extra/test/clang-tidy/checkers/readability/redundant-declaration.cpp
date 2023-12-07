// RUN: %check_clang_tidy %s readability-redundant-declaration %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {readability-redundant-declaration.IgnoreMacros: \
// RUN:                false}}"
//
// With -fms-compatibility and -DEXTERNINLINE, the extern inline shouldn't
// produce additional diagnostics, so same check suffix as before:
// RUN: %check_clang_tidy %s readability-redundant-declaration %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {readability-redundant-declaration.IgnoreMacros: \
// RUN:                false}}" -- -fms-compatibility -DEXTERNINLINE
//
// With -fno-ms-compatibility, DEXTERNINLINE causes additional output.
// (The leading ',' means "default checks in addition to NOMSCOMPAT checks.)
// RUN: %check_clang_tidy -check-suffix=,NOMSCOMPAT \
// RUN:   %s readability-redundant-declaration %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {readability-redundant-declaration.IgnoreMacros: \
// RUN:                false}}" -- -fno-ms-compatibility -DEXTERNINLINE

extern int Xyz;
extern int Xyz; // Xyz
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'Xyz' declaration [readability-redundant-declaration]
// CHECK-FIXES: {{^}}// Xyz{{$}}
int Xyz = 123;

extern int A;
extern int A, B;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'A' declaration
// CHECK-FIXES: {{^}}extern int A, B;{{$}}

extern int Buf[10];
extern int Buf[10]; // Buf[10]
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'Buf' declaration
// CHECK-FIXES: {{^}}// Buf[10]{{$}}

static int f();
static int f(); // f
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'f' declaration
// CHECK-FIXES: {{^}}// f{{$}}
static int f() {}

// Original check crashed for the code below.
namespace std {
typedef decltype(sizeof(0)) size_t;
}
void *operator new(std::size_t) __attribute__((__externally_visible__));
void *operator new[](std::size_t) __attribute__((__externally_visible__));

// Don't warn about static member definition.
struct C {
  static int I;
};
int C::I;

template <class T>
struct C2 {
  C2();
};

template <class T>
C2<T>::C2() = default;

void best_friend();

struct Friendly {
  friend void best_friend();
  friend void enemy();
};

void enemy();

template <typename>
struct TemplateFriendly {
  template <typename T>
  friend void generic_friend();
};

template <typename T>
void generic_friend() {}

TemplateFriendly<int> template_friendly;

template <typename>
struct TemplateFriendly2 {
  template <typename T>
  friend void generic_friend2() {}
};

template <typename T>
void generic_friend2();

void generic_friend_caller() {
  TemplateFriendly2<int> f;
  generic_friend2<int>();
}


namespace macros {
#define DECLARE(x) extern int x
#define DEFINE(x) extern int x; int x = 42
DECLARE(test);
DEFINE(test);
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant 'test' declaration
// CHECK-FIXES: {{^}}#define DECLARE(x) extern int x{{$}}
// CHECK-FIXES: {{^}}#define DEFINE(x) extern int x; int x = 42{{$}}
// CHECK-FIXES: {{^}}DECLARE(test);{{$}}
// CHECK-FIXES: {{^}}DEFINE(test);{{$}}

} // namespace macros

inline void g() {}

inline void g(); // g
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant 'g' declaration
// CHECK-FIXES: {{^}}// g{{$}}

#if defined(EXTERNINLINE)
extern inline void g(); // extern g
// CHECK-MESSAGES-NOMSCOMPAT: :[[@LINE-1]]:20: warning: redundant 'g' declaration
// CHECK-FIXES-NOMSCOMPAT: {{^}}// extern g{{$}}
#endif

// PR42068
extern "C" int externX;
int dummyBeforeBegin;extern "C" int externX;int dummyAfterEnd;
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: redundant 'externX' declaration [readability-redundant-declaration]
// CHECK-FIXES: {{^}}int dummyBeforeBegin;int dummyAfterEnd;{{$}}
