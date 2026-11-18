// Test that -fsanitize-cfi-icall-experimental-normalize-integers does not
// normalize C++ class vtable !type metadata and break WholeProgramDevirt:
//   1. Uniform ThinLTO with --lto-validate-all-vtables-have-type-infos still
//      matches vtable !type metadata against RTTI _ZTI symbols and
//      devirtualizes single-implementation virtual calls.
//   2. Mixed-TU ThinLTO (where only some TUs pass
//      -fsanitize-cfi-icall-experimental-normalize-integers) still sees
//      overriding subclasses across TUs and does not unsoundly devirtualize.

// REQUIRES: linux, lld-available, thinlto, cxxabi, (x86_64-target-arch || aarch64-target-arch)

// Static libunwind is built with -fno-rtti, so allow its internal UnwindCursor
// vtables under --lto-validate-all-vtables-have-type-infos.
// --- KCFI: uniform + --lto-validate-all-vtables-have-type-infos ---
// RUN: %clangxx -O2 -flto=thin -fno-split-lto-unit -fwhole-program-vtables \
// RUN:   -fsanitize=kcfi -fsanitize-cfi-icall-experimental-normalize-integers \
// RUN:   -DTU_A -c %s -o %t.kcfi.a.o
// RUN: %clangxx -O2 -flto=thin -fno-split-lto-unit -fwhole-program-vtables \
// RUN:   -fsanitize=kcfi -fsanitize-cfi-icall-experimental-normalize-integers \
// RUN:   -DTU_MAIN -c %s -o %t.kcfi.main.o
// RUN: %clangxx -O2 -flto=thin -fuse-ld=lld %t.kcfi.a.o %t.kcfi.main.o \
// RUN:   -o %t.kcfi.uniform -Wl,--lto-whole-program-visibility \
// RUN:   -Wl,--lto-validate-all-vtables-have-type-infos \
// RUN:   -Wl,--lto-known-safe-vtables='_ZTVN9libunwind*' \
// RUN:   -Wl,-mllvm,-pass-remarks=wholeprogramdevirt 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEVIRT
// RUN: %run %t.kcfi.uniform

// --- KCFI: mixed-TU (TU_B without normalize-integers defines overrides) ---
// RUN: %clangxx -O2 -flto=thin -fno-split-lto-unit -fwhole-program-vtables \
// RUN:   -DTU_B -c %s -o %t.plain.b.o
// RUN: %clangxx -O2 -flto=thin -fno-split-lto-unit -fwhole-program-vtables \
// RUN:   -fsanitize=kcfi -fsanitize-cfi-icall-experimental-normalize-integers \
// RUN:   -DTU_MAIN -DWITH_ALT -c %s -o %t.kcfi.main_alt.o
// RUN: %clangxx -O2 -flto=thin -fuse-ld=lld %t.kcfi.a.o %t.plain.b.o \
// RUN:   %t.kcfi.main_alt.o -o %t.kcfi.mixed -Wl,--lto-whole-program-visibility
// RUN: %run %t.kcfi.mixed

// --- cfi-icall: uniform + --lto-validate-all-vtables-have-type-infos ---
// RUN: %clangxx -O2 -flto=thin -fwhole-program-vtables \
// RUN:   -fsanitize=cfi-icall -fsanitize-trap=cfi-icall \
// RUN:   -fsanitize-cfi-icall-experimental-normalize-integers \
// RUN:   -DTU_A -c %s -o %t.cfi.a.o
// RUN: %clangxx -O2 -flto=thin -fwhole-program-vtables \
// RUN:   -fsanitize=cfi-icall -fsanitize-trap=cfi-icall \
// RUN:   -fsanitize-cfi-icall-experimental-normalize-integers \
// RUN:   -DTU_MAIN -c %s -o %t.cfi.main.o
// RUN: %clangxx -O2 -flto=thin -fuse-ld=lld -fsanitize=cfi-icall \
// RUN:   -fsanitize-trap=cfi-icall %t.cfi.a.o %t.cfi.main.o \
// RUN:   -o %t.cfi.uniform -Wl,--lto-whole-program-visibility \
// RUN:   -Wl,--lto-validate-all-vtables-have-type-infos \
// RUN:   -Wl,--lto-known-safe-vtables='_ZTVN9libunwind*' \
// RUN:   -Wl,-mllvm,-pass-remarks=wholeprogramdevirt 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEVIRT
// RUN: %run %t.cfi.uniform

// --- cfi-icall: mixed-TU (TU_B without normalize-integers defines overrides) ---
// RUN: %clangxx -O2 -flto=thin -fwhole-program-vtables \
// RUN:   -DTU_B -c %s -o %t.plain_split.b.o
// RUN: %clangxx -O2 -flto=thin -fwhole-program-vtables \
// RUN:   -fsanitize=cfi-icall -fsanitize-trap=cfi-icall \
// RUN:   -fsanitize-cfi-icall-experimental-normalize-integers \
// RUN:   -DTU_MAIN -DWITH_ALT -c %s -o %t.cfi.main_alt.o
// RUN: %clangxx -O2 -flto=thin -fuse-ld=lld -fsanitize=cfi-icall \
// RUN:   -fsanitize-trap=cfi-icall %t.cfi.a.o %t.plain_split.b.o \
// RUN:   %t.cfi.main_alt.o -o %t.cfi.mixed -Wl,--lto-whole-program-visibility
// RUN: %run %t.cfi.mixed

// DEVIRT-DAG: single-impl: devirtualized a call to _ZNK12_GLOBAL__N_17Primary3tagEv
// DEVIRT-DAG: single-impl: devirtualized a call to _ZNK12_GLOBAL__N_17Primary3valEi
// DEVIRT-DAG: single-impl: devirtualized a call to _ZNK3BoxIiE3getEi

struct Shape {
  virtual int tag() const = 0;
  virtual int val(int x) const = 0;
  virtual ~Shape() = default;
};

template <typename T> struct Box {
  virtual T get(T x) const { return x + 1; }
  virtual ~Box() = default;
};

Shape *make_primary();
Box<int> *make_primary_box();
Shape *make_alt();
Box<int> *make_alt_box();
int add_ten(int x);
int apply_icall(int (*fn)(int), int x);
int eval_shape(const Shape *s, const Box<int> *b, int x);

#if defined(TU_A)

namespace {
struct Primary : Shape {
  int tag() const override { return 10; }
  int val(int x) const override { return x + 100; }
};
} // namespace

template struct Box<int>;

Shape *make_primary() {
  static Primary p;
  return &p;
}

Box<int> *make_primary_box() {
  static Box<int> b;
  return &b;
}

int add_ten(int x) { return x + 10; }

__attribute__((noinline)) int apply_icall(int (*fn)(int), int x) {
  return fn(x);
}

__attribute__((noinline)) int eval_shape(const Shape *s, const Box<int> *b,
                                         int x) {
  return s->tag() + s->val(x) + b->get(x);
}

#elif defined(TU_B)

namespace {
struct Alt : Shape {
  int tag() const override { return 20; }
  int val(int x) const override { return x + 200; }
};

struct AltBox : Box<int> {
  int get(int x) const override { return x + 50; }
};
} // namespace

Shape *make_alt() {
  static Alt a;
  return &a;
}

Box<int> *make_alt_box() {
  static AltBox ab;
  return &ab;
}

#elif defined(TU_MAIN)

int main() {
  if (apply_icall(&add_ten, 5) != 15)
    return 1;
  if (eval_shape(make_primary(), make_primary_box(), 5) != 121)
    return 2;
#  if defined(WITH_ALT)
  if (eval_shape(make_alt(), make_alt_box(), 5) != 280)
    return 3;
#  endif
  return 0;
}

#endif
