// RUN: rm -rf %t.idx
// RUN: mkdir -p %t.o
// RUN: env CLANG_PROJECT_INDEX_PATH=%t.idx %clang -target x86_64-apple-macosx10.7 -arch x86_64 -mmacosx-version-min=10.7 -c %S/Inputs/test1.c -o %t.o/test1.o
// RUN: env CLANG_PROJECT_INDEX_PATH=%t.idx %clang -target x86_64-apple-macosx10.7 -arch x86_64 -mmacosx-version-min=10.7 -c %S/Inputs/test2.c -o %t.o/test2.o
// RUN: env CLANG_PROJECT_INDEX_PATH=%t.idx %clang -target x86_64-apple-macosx10.7 -arch x86_64 -mmacosx-version-min=10.7 -c %S/Inputs/test3.cpp -o %t.o/test3.o
// RUN: c-index-test core -aggregate-json %t.idx -o %t.json
// RUN: sed -e "s:%S::g" -e "s:%t.o::g" %t.json > %t.final.json
// RUN: diff -u %S/Inputs/json.c.json %t.final.json

// REQUIRES: shell
