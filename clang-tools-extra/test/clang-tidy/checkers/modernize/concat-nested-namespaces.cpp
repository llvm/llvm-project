// RUN: cp %S/Inputs/concat-nested-namespaces/modernize-concat-nested-namespaces.h %T/modernize-concat-nested-namespaces.h
// RUN: %check_clang_tidy -std=c++17 %s modernize-concat-nested-namespaces %t -- -header-filter=".*" -- -I %T
// RUN: FileCheck -input-file=%T/modernize-concat-nested-namespaces.h %S/Inputs/concat-nested-namespaces/modernize-concat-nested-namespaces.h -check-prefix=CHECK-FIXES
// Restore header file and re-run with c++20:
// RUN: cp %S/Inputs/concat-nested-namespaces/modernize-concat-nested-namespaces.h %T/modernize-concat-nested-namespaces.h
// RUN: %check_clang_tidy -std=c++20 %s modernize-concat-nested-namespaces %t -- -header-filter=".*" -- -I %T
// RUN: FileCheck -input-file=%T/modernize-concat-nested-namespaces.h %S/Inputs/concat-nested-namespaces/modernize-concat-nested-namespaces.h -check-prefix=CHECK-FIXES

#include "modernize-concat-nested-namespaces.h"
// CHECK-MESSAGES-DAG: modernize-concat-nested-namespaces.h:1:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]

namespace n1 {}

namespace n2 {
namespace n3 {
void t();
}
namespace n4 {
void t();
}
} // namespace n2

namespace n5 {
inline namespace inline_ns {
void t();
} // namespace inline_ns
} // namespace n5

namespace n6 {
namespace [[deprecated]] attr_ns {
void t();
} // namespace attr_ns
} // namespace n6

namespace n7 {
void t();

namespace n8 {
void t();
}
} // namespace n7

namespace n9 {
namespace n10 {
// CHECK-MESSAGES-DAG: :[[@LINE-2]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
// CHECK-FIXES: namespace n9::n10
void t();
} // namespace n10
} // namespace n9
// CHECK-FIXES: }

namespace n11 {
namespace n12 {
// CHECK-MESSAGES-DAG: :[[@LINE-2]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
// CHECK-FIXES: namespace n11::n12
namespace n13 {
void t();
}
namespace n14 {
void t();
}
} // namespace n12
} // namespace n11
// CHECK-FIXES: }

namespace n15 {
namespace n16 {
void t();
}

inline namespace n17 {
void t();
}

namespace n18 {
namespace n19 {
namespace n20 {
// CHECK-MESSAGES-DAG: :[[@LINE-3]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
// CHECK-FIXES: namespace n18::n19::n20
void t();
} // namespace n20
} // namespace n19
} // namespace n18
// CHECK-FIXES: }

namespace n21 {
void t();
}
} // namespace n15

namespace n22 {
namespace {
void t();
}
} // namespace n22

namespace n23 {
namespace {
namespace n24 {
namespace n25 {
// CHECK-MESSAGES-DAG: :[[@LINE-2]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
// CHECK-FIXES: namespace n24::n25
void t();
} // namespace n25
} // namespace n24
// CHECK-FIXES: }
} // namespace
} // namespace n23

namespace n26::n27 {
namespace n28 {
namespace n29::n30 {
// CHECK-MESSAGES-DAG: :[[@LINE-3]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
// CHECK-FIXES: namespace n26::n27::n28::n29::n30 {
void t() {}
} // namespace n29::n30
} // namespace n28
} // namespace n26::n27
// CHECK-FIXES: }

namespace n31 {
namespace n32 {}
// CHECK-MESSAGES-DAG: :[[@LINE-2]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
} // namespace n31
// CHECK-FIXES-EMPTY

namespace n33 {
namespace n34 {
namespace n35 {}
// CHECK-MESSAGES-DAG: :[[@LINE-2]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
} // namespace n34
// CHECK-FIXES-EMPTY
namespace n36 {
void t();
}
} // namespace n33

namespace n37::n38 {
void t();
}

#define IEXIST
namespace n39 {
namespace n40 {
// CHECK-MESSAGES-DAG: :[[@LINE-2]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
// CHECK-FIXES: namespace n39::n40
#ifdef IEXIST
void t() {}
#endif
} // namespace n40
} // namespace n39
// CHECK-FIXES: } // namespace n39::n40

namespace n41 {
namespace n42 {
// CHECK-MESSAGES-DAG: :[[@LINE-2]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
// CHECK-FIXES: namespace n41::n42
#ifdef IDONTEXIST
void t() {}
#endif
} // namespace n42
} // namespace n41
// CHECK-FIXES: } // namespace n41::n42


// CHECK-MESSAGES-DAG: :[[@LINE+1]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
namespace n43 {
#define N43_INNER
namespace n44 {
void foo() {}
} // namespace n44
#undef N43_INNER
} // namespace n43
// CHECK-FIXES: #define N43_INNER
// CHECK-FIXES: namespace n43::n44 {
// CHECK-FIXES: } // namespace n43::n44
// CHECK-FIXES: #undef N43_INNER

// CHECK-MESSAGES-DAG: :[[@LINE+1]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
namespace n45{
#define N45_INNER
namespace n46
{
#pragma clang diagnostic push
namespace n47 {
void foo() {}
} // namespace n47
#pragma clang diagnostic pop
} //namespace n46
#undef N45_INNER
} //namespace n45
// CHECK-FIXES: #define N45_INNER
// CHECK-FIXES: #pragma clang diagnostic push
// CHECK-FIXES: namespace n45::n46::n47 {
// CHECK-FIXES: } // namespace n45::n46::n47
// CHECK-FIXES: #pragma clang diagnostic pop
// CHECK-FIXES: #undef N45_INNER

// CHECK-MESSAGES-DAG: :[[@LINE+1]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
namespace avoid_add_close_comment {
namespace inner {
void foo() {}
}
}
// CHECK-FIXES: namespace avoid_add_close_comment::inner {
// CHECK-FIXES-NOT: } // namespace avoid_add_close_comment::inner

// CHECK-MESSAGES-DAG: :[[@LINE+1]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
namespace avoid_change_close_comment {
namespace inner {
void foo() {}
} // namespace inner and other comments
} // namespace avoid_change_close_comment and other comments
// CHECK-FIXES: namespace avoid_change_close_comment::inner {
// CHECK-FIXES-NOT: } // namespace avoid_add_close_comment::inner

namespace /*::*/ comment_colon_1 {
void foo() {}
} // namespace comment_colon_1
// CHECK-FIXES: namespace /*::*/ comment_colon_1 {

// CHECK-MESSAGES-DAG: :[[@LINE+1]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
namespace /*::*/ comment_colon_2 {
namespace comment_colon_2 {
void foo() {}
} // namespace comment_colon_2
} // namespace comment_colon_2

int main() {
  n26::n27::n28::n29::n30::t();
#ifdef IEXIST
  n39::n40::t();
#endif

#ifdef IDONTEXIST
  n41::n42::t();
#endif

  return 0;
}
