// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -o - %s -fsanitize=signed-integer-overflow -fsanitize-skip-hot-cutoff=signed-integer-overflow=0.000001 -O3 -mllvm -lower-allow-check-random-rate=1 -Rpass=lower-allow-check -Rpass-missed=lower-allow-check -fno-inline 2>&1 | FileCheck %s --check-prefixes=NOINL --implicit-check-not="remark:"
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -o - %s -fsanitize=signed-integer-overflow -fsanitize-skip-hot-cutoff=signed-integer-overflow=0.000001 -O3 -mllvm -lower-allow-check-random-rate=1 -Rpass=lower-allow-check -Rpass-missed=lower-allow-check 2>&1 | FileCheck %s --check-prefixes=INLINE --implicit-check-not="remark:"
//
// -ubsan-guard-checks is deprecated and will be removed in the future;
// use -fsanitize-skip-hot-cutoff, as shown above.
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -o - %s -fsanitize=signed-integer-overflow -mllvm -ubsan-guard-checks -O3 -mllvm -lower-allow-check-random-rate=1 -Rpass=lower-allow-check -Rpass-missed=lower-allow-check -fno-inline 2>&1 | FileCheck %s --check-prefixes=NOINL --implicit-check-not="remark:"
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -o - %s -fsanitize=signed-integer-overflow -mllvm -ubsan-guard-checks -O3 -mllvm -lower-allow-check-random-rate=1 -Rpass=lower-allow-check -Rpass-missed=lower-allow-check 2>&1 | FileCheck %s --check-prefixes=INLINE --implicit-check-not="remark:"

int get();
void set(int x);

// We will only make decision in the `overflow` function.
// NOINL-COUNT-1: remark: Allowed check:

// We will make decision on every inline.
// INLINE-COUNT-5: remark: Allowed check:

static void overflow() {
  set(get() + get());
}

void test() {
  overflow();
  overflow();
  overflow();
  overflow();
  overflow();
}
