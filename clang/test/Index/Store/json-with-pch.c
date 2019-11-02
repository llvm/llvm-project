// RUN: rm -rf %t.idx
// RUN: %clang -target x86_64-apple-darwin -arch x86_64 -mmacosx-version-min=10.7 -x c-header %S/Inputs/head.h -o %t.h.pch -index-store-path %t.idx
// RUN: %clang -target x86_64-apple-darwin -arch x86_64 -mmacosx-version-min=10.7 -c %s -o %t.o -index-store-path %t.idx -include %t.h -Werror
// RUN: c-index-test core -aggregate-json %t.idx -o %t.json
// RUN: sed -e "s:%S::g" -e "s:%T::g" %t.json > %t.final.json
// RUN: diff -u %s.json %t.final.json

int main() {
  test1_func();
}

// REQUIRES: shell
