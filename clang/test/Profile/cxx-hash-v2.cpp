// REQUIRES: shell

// Check that all of the hashes in this file are unique (i.e, that none of the
// profiles for these functions are mutually interchangeable).
//
// RUN: llvm-profdata show -all-functions %S/Inputs/cxx-hash-v2.profdata.v5 | grep "Hash: 0x" | sort > %t.hashes
// RUN: uniq %t.hashes > %t.hashes.unique
// RUN: diff %t.hashes %t.hashes.unique

// RUN: llvm-profdata merge %S/Inputs/cxx-hash-v2.proftext -o %t.profdata
// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -triple x86_64-apple-macosx10.9 -main-file-name cxx-hash-v2.mm %s -o /dev/null -emit-llvm -fprofile-instrument-use-path=%t.profdata 2>&1 | FileCheck %s -allow-empty
// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -triple x86_64-apple-macosx10.9 -main-file-name cxx-hash-v2.mm %s -o /dev/null -emit-llvm -fprofile-instrument-use-path=%S/Inputs/cxx-hash-v2.profdata.v5 2>&1 | FileCheck %s -allow-empty

// CHECK-NOT: warning: profile data may be out of date

int x;
int arr[1] = {0};

void loop_after_if_else() {
  if (1)
    x = 1;
  else
    x = 2;
  while (0)
    ++x;
}

void loop_in_then_block() {
  if (1) {
    while (0)
      ++x;
  } else {
    x = 2;
  }
}

void loop_in_else_block() {
  if (1) {
    x = 1;
  } else {
    while (0)
      ++x;
  }
}

void if_inside_of_for() {
  for (x = 0; x < 0; ++x) {
    x = 1;
    if (1)
      x = 2;
  }
}

void if_outside_of_for() {
  for (x = 0; x < 0; ++x)
    x = 1;
  if (1)
    x = 2;
}

void if_inside_of_while() {
  while (0) {
    x = 1;
    if (1)
      x = 2;
  }
}

void if_outside_of_while() {
  while (0)
    x = 1;
  if (1)
    x = 2;
}

void nested_dos() {
  do {
    do {
      ++x;
    } while (0);
  } while (0);
}

void consecutive_dos() {
  do {
  } while (0);
  do {
    ++x;
  } while (0);
}

void loop_empty() {
  for (x = 0; x < 5; ++x) {}
}

void loop_return() {
  for (x = 0; x < 5; ++x)
    return;
}

void loop_continue() {
  for (x = 0; x < 5; ++x)
    continue;
}

void loop_break() {
  for (x = 0; x < 5; ++x)
    break;
}

void no_gotos() {
  static void *dispatch[] = {&&done};
  x = 0;
done:
  ++x;
}

void direct_goto() {
  static void *dispatch[] = {&&done};
  x = 0;
  goto done;
done:
  ++x;
}

void indirect_goto() {
  static void *dispatch[] = {&&done};
  x = 0;
  goto *dispatch[x];
done:
  ++x;
}

void nested_for_ranges() {
  for (int a : arr)
    for (int b : arr)
      ++x;
}

void consecutive_for_ranges() {
  for (int a : arr) {}
  for (int b : arr)
    ++x;
}

void nested_try_catch() {
  try {
    try {
      ++x;
    } catch (...) {}
  } catch (...) {}
}

void consecutive_try_catch() {
  try {} catch (...) {}
  try {
    ++x;
  } catch (...) {}
}

void no_throw() {}

void has_throw() {
  throw 0;
}

void single_lnot() {
  if (!x) {}
}

void double_lnot() {
  if (!!x) {}
}

int main() {
  return 0;
}
