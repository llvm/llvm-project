// RUN: %check_clang_tidy %s google-readability-namespace-comments %t

namespace n1 {
namespace /* a comment */ n2 /* another comment */ {


void f(); // So that the namespace isn't empty.


// CHECK-MESSAGES: :[[@LINE+4]]:1: warning: namespace 'n2' not terminated with a closing comment [google-readability-namespace-comments]
// CHECK-MESSAGES: :[[@LINE-7]]:27: note: namespace 'n2' starts here
// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: namespace 'n1' not terminated with
// CHECK-MESSAGES: :[[@LINE-10]]:11: note: namespace 'n1' starts here
}}
// CHECK-FIXES: }  // namespace n2
// CHECK-FIXES: }  // namespace n1

#define MACRO macro_expansion
namespace MACRO {
void f(); // So that the namespace isn't empty.
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// CHECK-MESSAGES: :[[@LINE+2]]:1: warning: namespace 'MACRO' not terminated with
// CHECK-MESSAGES: :[[@LINE-10]]:11: note: namespace 'MACRO' starts here
}
// CHECK-FIXES: }  // namespace MACRO

namespace macro_expansion {
void ff(); // So that the namespace isn't empty.
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// CHECK-MESSAGES: :[[@LINE+2]]:1: warning: namespace 'macro_expansion' not terminated with
// CHECK-MESSAGES: :[[@LINE-10]]:11: note: namespace 'macro_expansion' starts here
}
// CHECK-FIXES: }  // namespace macro_expansion

namespace [[deprecated("foo")]] namespace_with_attr {
inline namespace inline_namespace {
void g();
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// CHECK-MESSAGES: :[[@LINE+2]]:1: warning: namespace 'inline_namespace' not terminated with
// CHECK-MESSAGES: :[[@LINE-10]]:18: note: namespace 'inline_namespace' starts here
}
// CHECK-FIXES: }  // namespace inline_namespace
// CHECK-MESSAGES: :[[@LINE+2]]:1: warning: namespace 'namespace_with_attr' not terminated with
// CHECK-MESSAGES: :[[@LINE-15]]:33: note: namespace 'namespace_with_attr' starts here
}
// CHECK-FIXES: }  // namespace namespace_with_attr

namespace [[deprecated]] {
void h();
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// CHECK-MESSAGES: :[[@LINE+2]]:1: warning: anonymous namespace not terminated with
// CHECK-MESSAGES: :[[@LINE-10]]:26: note: anonymous namespace starts here
}
// CHECK-FIXES: }  // namespace{{$}}

namespace [[]] {
void hh();
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// CHECK-MESSAGES: :[[@LINE+2]]:1: warning: anonymous namespace not terminated with
// CHECK-MESSAGES: :[[@LINE-10]]:16: note: anonymous namespace starts here
}
// CHECK-FIXES: }  // namespace{{$}}

namespace short1 {
namespace short2 {
// Namespaces covering 10 lines or fewer are exempt from this rule.





}
}

namespace n3 {









}; // namespace n3
