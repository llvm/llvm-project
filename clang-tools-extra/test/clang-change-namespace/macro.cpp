// RUN: cp %S/macro.cpp %T/macro.cpp
// RUN: echo "#define USING using na::nc::X" > %T/macro.h
//
// RUN: clang-change-namespace -old_namespace "na::nb" -new_namespace "x::y" --file_pattern "macro.cpp" --i %T/macro.cpp --
// RUN: FileCheck -input-file=%T/macro.cpp -check-prefix=CHECK-CC %s
// RUN: FileCheck -input-file=%T/macro.h -check-prefix=CHECK-HEADER %s
//
// RUN: cp %S/macro.cpp %T/macro.cpp
// RUN: echo "#define USING using na::nc::X" > %T/macro.h
// RUN: clang-change-namespace -old_namespace "na::nb" -new_namespace "x::y" --file_pattern ".*" --i %T/macro.cpp --
// RUN: FileCheck -input-file=%T/macro.cpp -check-prefix=CHECK-CC %s
// RUN: FileCheck -input-file=%T/macro.h -check-prefix=CHECK-CHANGED-HEADER %s
#include "macro.h"
namespace na { namespace nc { class X{}; } }

namespace na {
namespace nb {
USING;
}
}
// CHECK-CC: namespace x {
// CHECK-CC: namespace y {
// CHECK-CC: USING;
// CHECK-CC: } // namespace y
// CHECK-CC: } // namespace x

// CHECK-HEADER: #define USING using na::nc::X

// CHECK-CHANGED-HEADER: #define USING using ::na::nc::X
