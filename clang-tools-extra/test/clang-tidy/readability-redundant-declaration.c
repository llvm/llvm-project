// RUN: %check_clang_tidy %s readability-redundant-declaration %t

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

inline void g() {}

inline void g();
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant 'g' declaration

// OK: Needed to emit an external definition.
extern inline void g();
