// Verify ubsan doesn't emit checks for blacklisted functions and files
// RUN: echo "fun:hash" > %t-func.blacklist
// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t-file.blacklist
// RUN: %clang_cc1 -fsanitize=unsigned-integer-overflow -emit-llvm %s -o - | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cc1 -fsanitize=unsigned-integer-overflow -fsanitize-blacklist=%t-func.blacklist -emit-llvm %s -o - | FileCheck %s --check-prefix=FUNC
// RUN: %clang_cc1 -fsanitize=unsigned-integer-overflow -fsanitize-blacklist=%t-file.blacklist -emit-llvm %s -o - | FileCheck %s --check-prefix=FILE

unsigned i;

// DEFAULT: @hash
// FUNC: @hash
// FILE: @hash
unsigned hash() {
// DEFAULT: call {{.*}}void @__ubsan
// FUNC-NOT: call {{.*}}void @__ubsan
// FILE-NOT: call {{.*}}void @__ubsan
  return i * 37;
}

// DEFAULT: @add
// FUNC: @add
// FILE: @add
unsigned add() {
// DEFAULT: call {{.*}}void @__ubsan
// FUNC: call {{.*}}void @__ubsan
// FILE-NOT: call {{.*}}void @__ubsan
  return i + 1;
}
