// RUN: not %clang -o /dev/null -v -fxray-instrument -c %s
// XFAIL: amd64-, x86_64-, x86_64h-, arm
// REQUIRES: linux
typedef int a;
