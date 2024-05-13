// RUN: clang-tidy %s -checks='-*,google-build-namespaces,google-build-using-namespace' -header-filter='.*' -- | FileCheck %s -implicit-check-not="{{warning|error}}:"
#include "Inputs/google-namespaces.h"
// CHECK: warning: do not use unnamed namespaces in header files [google-build-namespaces]

using namespace spaaaace;
// CHECK: :[[@LINE-1]]:1: warning: do not use namespace using-directives; use using-declarations instead [google-build-using-namespace]

using spaaaace::core; // no-warning

namespace std {
inline namespace literals {
inline namespace chrono_literals {
}
inline namespace complex_literals {
}
inline namespace string_literals {
}
}
}

using namespace std::chrono_literals;            // no-warning
using namespace std::complex_literals;           // no-warning
using namespace std::literals;                   // no-warning
using namespace std::literals::chrono_literals;  // no-warning
using namespace std::literals::complex_literals; // no-warning
using namespace std::literals::string_literals;  // no-warning
using namespace std::string_literals;            // no-warning

namespace literals {}

using namespace literals;
// CHECK: :[[@LINE-1]]:1: warning: do not use namespace using-directives; use using-declarations instead [google-build-using-namespace]

namespace foo {
inline namespace literals {
inline namespace bar_literals {}
}
}

using namespace foo::literals;
// CHECK: :[[@LINE-1]]:1: warning: do not use namespace using-directives; use using-declarations instead [google-build-using-namespace]

using namespace foo::bar_literals;
// CHECK: :[[@LINE-1]]:1: warning: do not use namespace using-directives; use using-declarations instead [google-build-using-namespace]

using namespace foo::literals::bar_literals;
// CHECK: :[[@LINE-1]]:1: warning: do not use namespace using-directives; use using-declarations instead [google-build-using-namespace]

namespace foo_literals {}

using namespace foo_literals;
// CHECK: :[[@LINE-1]]:1: warning: do not use namespace using-directives; use using-declarations instead [google-build-using-namespace]
