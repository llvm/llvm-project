// RUN: env CLANG_SPAWN_CC1= %clang -S %s -o /dev/null
// RUN: env CLANG_SPAWN_CC1=0 %clang -S %s -o /dev/null
// RUN: env CLANG_SPAWN_CC1=1 %clang -S %s -o /dev/null
// RUN: env CLANG_SPAWN_CC1=test not %clang -S %s -o /dev/null
