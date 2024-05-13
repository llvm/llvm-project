// RUN: %check_clang_tidy %s readability-redundant-declaration %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {readability-redundant-declaration.IgnoreMacros: \
// RUN:                true}}"

extern int Xyz;
extern int Xyz; // Xyz
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'Xyz' declaration [readability-redundant-declaration]
// CHECK-FIXES: {{^}}// Xyz{{$}}

namespace macros {
#define DECLARE(x) extern int x
#define DEFINE(x) extern int x; int x = 42
DECLARE(test);
DEFINE(test);
// CHECK-FIXES: {{^}}#define DECLARE(x) extern int x{{$}}
// CHECK-FIXES: {{^}}#define DEFINE(x) extern int x; int x = 42{{$}}
// CHECK-FIXES: {{^}}DECLARE(test);{{$}}
// CHECK-FIXES: {{^}}DEFINE(test);{{$}}

} // namespace macros
