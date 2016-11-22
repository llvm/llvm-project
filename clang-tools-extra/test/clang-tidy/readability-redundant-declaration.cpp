// RUN: %check_clang_tidy %s readability-redundant-declaration %t

extern int Xyz;
extern int Xyz;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'Xyz' declaration [readability-redundant-declaration]
// CHECK-FIXES: {{^}}{{$}}
int Xyz = 123;

extern int A;
extern int A, B;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'A' declaration
// CHECK-FIXES: {{^}}extern int A, B;{{$}}

extern int Buf[10];
extern int Buf[10];
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'Buf' declaration
// CHECK-FIXES: {{^}}{{$}}

static int f();
static int f();
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'f' declaration
// CHECK-FIXES: {{^}}{{$}}
static int f() {}

// Original check crashed for the code below.
namespace std {
  typedef decltype(sizeof(0)) size_t;
}
void* operator new(std::size_t) __attribute__((__externally_visible__));
void* operator new[](std::size_t) __attribute__((__externally_visible__));
