// RUN: %check_clang_tidy %s google-readability-namespace-comments %t -std=c++20 \
// RUN:   '-config={CheckOptions: { \
// RUN:     google-readability-namespace-comments.AllowOmittingNamespaceComments: true, \
// RUN:     google-readability-namespace-comments.ShortNamespaceLines: 0, \
// RUN:   }}'

// accept if namespace comments are fully omitted
namespace n1::n2 {
namespace /*comment1*/n3/*comment2*/::/*comment3*/inline/*comment4*/n4/*comment5*/ {
void f();
}}

namespace n5::inline n6 {
void f();
}

namespace n7::inline n8 {
void f();
}

// accept if namespace comments are partly omitted (e.g. only for nested namespace)
namespace n1::n2 {
namespace n3::n4 {
void f();
}
} // namespace n1::n2

// fail if namespace comment is different than expected
namespace n9::inline n10 {
void f();
} // namespace n9::n10
// CHECK-MESSAGES: :[[@LINE-1]]:2: warning: namespace 'n9::inline n10' ends with a comment that refers to a wrong namespace 'n9::n10' [google-readability-namespace-comments]
