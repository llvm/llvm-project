// RUN: %clang_cc1 -no-opaque-pointers "-triple" "arm64-windows" "-funwind-tables=2" "-fms-compatibility" -emit-llvm -O1 -disable-llvm-passes %s -o - | FileCheck %s
// NOTE: we're passing "-O1 -disable-llvm-passes" to avoid adding optnone and noinline everywhere.

# 0 "" 3
#define a(b, c) d() & b
#define f(c) a(e(0, 0, #c).b(), )

struct e {
  e(int, int, char *);
  int b();
};

struct d {
  void operator&(int);
};

struct h;

struct i {
  h *operator->();
  h &operator*() { f(); }
};

typedef int g;

struct h {
  void ad();
};

g aq(h j, g k, int, int) {
  if (k)
    return;
  j.ad();
}

// Check for the uwtable attribute on the filter funclet.
// CHECK: define internal noundef i32 @"?filt$0@0@at@@"(i8* noundef %exception_pointers, i8* noundef %frame_pointer) #[[MD:[0-9]+]]
// CHECK: attributes #[[MD]] = { nounwind uwtable

void at() {
  i ar;

  __try {
    ar->ad();
  } __except (aq(*ar, _exception_code(), 0, 0)) {
  }

}
