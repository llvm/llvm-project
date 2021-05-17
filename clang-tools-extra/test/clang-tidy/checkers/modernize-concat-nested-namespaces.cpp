// RUN: cp %S/Inputs/modernize-concat-nested-namespaces/modernize-concat-nested-namespaces.h %T/modernize-concat-nested-namespaces.h
// RUN: %check_clang_tidy -std=c++17 %s modernize-concat-nested-namespaces %t -- -header-filter=".*" -- -I %T
// RUN: FileCheck -input-file=%T/modernize-concat-nested-namespaces.h %S/Inputs/modernize-concat-nested-namespaces/modernize-concat-nested-namespaces.h -check-prefix=CHECK-FIXES
// Restore header file and re-run with c++20:
// RUN: cp %S/Inputs/modernize-concat-nested-namespaces/modernize-concat-nested-namespaces.h %T/modernize-concat-nested-namespaces.h
// RUN: %check_clang_tidy -std=c++20 %s modernize-concat-nested-namespaces %t -- -header-filter=".*" -- -I %T
// RUN: FileCheck -input-file=%T/modernize-concat-nested-namespaces.h %S/Inputs/modernize-concat-nested-namespaces/modernize-concat-nested-namespaces.h -check-prefix=CHECK-FIXES

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
inline namespace n6 {
void t();
}
} // namespace n5

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
// CHECK-FIXES: namespace n26::n27::n28::n29::n30
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
// CHECK-FIXES: }

namespace n41 {
namespace n42 {
// CHECK-MESSAGES-DAG: :[[@LINE-2]]:1: warning: nested namespaces can be concatenated [modernize-concat-nested-namespaces]
// CHECK-FIXES: namespace n41::n42
#ifdef IDONTEXIST
void t() {}
#endif
} // namespace n42
} // namespace n41
// CHECK-FIXES: }

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
