// RUN: rm -rf %t.idx %t.mcp
// RUN: %clang -target x86_64-apple-darwin -arch x86_64 -mmacosx-version-min=10.7 -c %s -o %t.o -index-store-path %t.idx -fmodules -fmodules-cache-path=%t.mcp -Xclang -fdisable-module-hash -I %S/Inputs/module
// RUN: c-index-test core -aggregate-json %t.idx -o %t.json
// RUN: sed -e "s:%S::g" -e "s:%T::g" %t.json > %t.final.json
// RUN: diff -u %s.json %t.final.json

@import ModDep;

// REQUIRES: shell
