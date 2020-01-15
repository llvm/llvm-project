// RUN: env CLANG_SPAWN_CC1= %clang -c %s -o /dev/null
// RUN: env CLANG_SPAWN_CC1=0 %clang -c %s -o /dev/null
// RUN: env CLANG_SPAWN_CC1=1 %clang -c %s -o /dev/null
// RUN: env CLANG_SPAWN_CC1=test not %clang -c %s -o /dev/null
