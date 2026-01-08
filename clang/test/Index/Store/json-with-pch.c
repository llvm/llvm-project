int main() {
  test1_func();
}

// UNSUPPORTED: system-windows

// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: %clang -target x86_64-apple-darwin -arch x86_64 -mmacosx-version-min=10.7 -x c-header %S/Inputs/head.h -o %t/head.h.pch -index-store-path %t/idx
// RUN: %clang -target x86_64-apple-darwin -arch x86_64 -mmacosx-version-min=10.7 -c %s -o %t/head.o -index-store-path %t/idx -include %t/head.h -Werror
// RUN: c-index-test core -aggregate-json %t/idx -o %t/output.json
// RUN: sed -e "s:%S::g" -e "s:%t::g" %t/output.json > %t/final.json
// RUN: diff -u %s.json %t/final.json
