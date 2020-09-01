// RUN: not %clang -o /dev/null -v -fxray-instrument -c %s
// XFAIL: -linux-, -freebsd, x86_64-apple-darwin, x86_64-apple-macos
// REQUIRES: amd64 || x86_64 || x86_64h || arm || aarch64 || arm64
typedef int a;
