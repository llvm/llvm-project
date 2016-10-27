// RUN: not %clang -o /dev/null -v -fxray-instrument -c %s
// XFAIL: -linux-
// REQUIRES-ANY: amd64, x86_64, x86_64h, arm
typedef int a;
