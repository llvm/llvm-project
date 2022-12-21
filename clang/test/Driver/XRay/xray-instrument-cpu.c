// RUN: not %clang -o /dev/null -v -fxray-instrument -c %s
// XFAIL: target={{(amd64|x86_64|x86_64h|powerpc64le)-.*}}
// XFAIL: target={{(arm|aarch64|arm64|mips|mipsel|mips64|mips64el)-.*}}
// REQUIRES: linux
typedef int a;
