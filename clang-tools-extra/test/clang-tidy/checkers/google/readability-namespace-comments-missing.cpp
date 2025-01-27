// RUN: %check_clang_tidy %s google-readability-namespace-comments %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     google-readability-namespace-comments.AllowNoNamespaceComments: true, \
// RUN:   }}'

namespace n1 {
namespace /* a comment */ n2 /* another comment */ {


void f(); // So that the namespace isn't empty.


}}

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
}

namespace macro_expansion {
void ff(); // So that the namespace isn't empty.
// 1
// 2
// 3
// 4
// 5
// 6
// 7
}

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
}
}

namespace [[]] {
void hh();
// 1
// 2
// 3
// 4
// 5
// 6
// 7
}

namespace short1 {
namespace short2 {
// Namespaces covering 10 lines or fewer
}
}

namespace n3 {









} // namespace n3

namespace n4 {
void hh();
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: namespace 'n4' ends with a comment that refers to a wrong namespace 'n5' [google-readability-namespace-comments]
}; // namespace n5
// CHECK-FIXES: }  // namespace n4
