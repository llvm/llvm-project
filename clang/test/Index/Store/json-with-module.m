@import ModDep;

// UNSUPPORTED: system-windows

// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: %clang -target x86_64-apple-darwin -arch x86_64 -mmacosx-version-min=10.7 -c %s -o %t/json.o -index-store-path %t/idx -fmodules -fmodules-cache-path=%t/mcp -Xclang -fdisable-module-hash -I %S/Inputs/module
// RUN: c-index-test core -aggregate-json %t/idx -o %t/output.json
// RUN: sed -e "s:%S::g" -e "s:%t::g" %t/output.json > %t/final.json
// RUN: diff -u %s.json %t/final.json
