// Make sure it doesn't crash.
// RUN: %clang -target x86_64-apple-macosx10.7 -S %s -o %t.s
// RUN: env CLANG_PROJECT_INDEX_PATH=%t.idx %clang -target x86_64-apple-macosx10.7 -c %t.s -o %t.o
