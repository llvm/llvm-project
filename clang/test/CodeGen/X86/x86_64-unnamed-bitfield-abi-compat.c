// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | \
// RUN:   FileCheck %s -check-prefix=NEW
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fclang-abi-compat=23 %s -o - | \
// RUN:   FileCheck %s -check-prefix=COMPAT
// RUN: %clang_cc1 -triple x86_64-scei-ps4 -emit-llvm %s -o - | \
// RUN:   FileCheck %s -check-prefix=COMPAT
// RUN: %clang_cc1 -triple x86_64-sie-ps5 -emit-llvm %s -o - | \
// RUN:   FileCheck %s -check-prefix=COMPAT

// A non-zero-width unnamed bit-field classifies the eightbyte it occupies as
// INTEGER like a named one, matching GCC, so this struct travels in two integer
// registers. Under -fclang-abi-compat=23 (and on PlayStation) the unnamed
// bit-field is padding, so the low eightbyte stays NO_CLASS and only the high
// eightbyte is passed.
struct s {
  long : 64;
  long a;
};

// NEW-LABEL:    define{{.*}} { i64, i64 } @get()
// COMPAT-LABEL: define{{.*}} i64 @get()
struct s get(void) {
  return (struct s){0};
}

// NEW-LABEL:    define{{.*}} void @put(i64 %a.coerce0, i64 %a.coerce1)
// COMPAT-LABEL: define{{.*}} void @put(i64 %a.coerce)
void put(struct s a) {
}

// Note: -fclang-abi-compat=23 faithfully reproduces Clang 23, so it also
// reproduces Clang 23's crash on a run of __int128 bit-fields (skipping the
// unnamed field leaves half of the i128 access unit unclassified). That shape
// is deliberately not exercised here; the corrected classification is covered
// in x86_64-arguments.c.
