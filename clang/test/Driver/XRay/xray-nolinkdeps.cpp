// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fxray-instrument -fxray-link-deps -fno-xray-link-deps %s \
// RUN:     2>&1 | FileCheck --check-prefix DISABLE %s
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fxray-instrument %s \
// RUN:     2>&1 | FileCheck --check-prefix ENABLE %s
// ENABLE:      "--whole-archive" "{{.*}}clang_rt.xray{{.*}}"--no-whole-archive"
// DISABLE-NOT: clang_rt.xray
