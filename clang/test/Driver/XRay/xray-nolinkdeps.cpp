// RUN: %clang -v -o /dev/null -fxray-instrument -fnoxray-link-deps %s -### \
// RUN:     2>&1 | FileCheck --check-prefix DISABLE %s
// RUN: %clang -v -o /dev/null -fxray-instrument -fxray-link-deps %s -### \
// RUN:     2>&1 | FileCheck --check-prefix ENABLE %s
// ENABLE: clang_rt.xray
// DISABLE-NOT: clang_rt.xray
// REQUIRES-ANY: linux, freebsd
// REQUIRES-ANY: amd64, x86_64, x86_64h, arm, aarch64, arm64
