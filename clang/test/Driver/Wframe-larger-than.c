// RUN: %clang -Wframe-larger-than=42 \
// RUN:   -v -E - < /dev/null 2>&1 | FileCheck %s --check-prefix=ENABLE
// RUN: %clang -Wframe-larger-than=42 -Wno-frame-larger-than= \
// RUN:   -v -E - < /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLE
// RUN: %clang -Wframe-larger-than=42 -Wno-frame-larger-than= -Wframe-larger-than=43 \
// RUN:   -v -E - < /dev/null 2>&1 | FileCheck %s --check-prefix=REENABLE
//
// TODO: we might want to look into being able to disable, then re-enable this
// warning properly. We could have the driver turn -Wframe-larger-than=X into
// -Wframe-larger-than -fwarn-stack-size=X.  Also, we should support
// -Wno-frame-larger-than (no = suffix) like GCC.

// ENABLE: cc1 {{.*}} -fwarn-stack-size=42
// DISABLE: cc1 {{.*}} -fwarn-stack-size=42 {{.*}} -Wno-frame-larger-than=
// REENABLE: cc1 {{.*}} -fwarn-stack-size=43 {{.*}} -Wno-frame-larger-than=
