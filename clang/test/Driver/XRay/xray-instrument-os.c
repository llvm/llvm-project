// RUN: not %clang -o /dev/null -v -fxray-instrument -c %s
// XFAIL: target={{.*-(linux|freebsd).*}}, target=x86_64-apple-{{(darwin|macos).*}}
// REQUIRES: target={{(amd64|x86_64|x86_64h|arm|aarch64|arm64)-.*}}
typedef int a;
