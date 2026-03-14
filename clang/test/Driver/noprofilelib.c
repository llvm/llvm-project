// RUN: %clang -target i686-pc-linux-gnu -### %s 2>&1 \
// RUN:     -fprofile-generate -noprofilelib | FileCheck %s
// RUN: %clang -target i686-pc-linux-gnu -### %s 2>&1 \
// RUN:     -fprofile-instr-generate -noprofilelib | FileCheck %s
// CHECK-NOT: clang_rt.profile
