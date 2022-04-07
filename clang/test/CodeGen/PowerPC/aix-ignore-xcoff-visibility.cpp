// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-unknown-aix -emit-llvm -o - -x c++ %s  | \
// RUN: FileCheck -check-prefix=NOVISIBILITY-IR %s

// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-unknown-aix -emit-llvm -round-trip-args -o - -x c++ %s  | \
// RUN: FileCheck -check-prefix=NOVISIBILITY-IR %s

// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-unknown-aix -mignore-xcoff-visibility -fvisibility default -emit-llvm -o - -x c++ %s  | \
// RUN: FileCheck -check-prefix=NOVISIBILITY-IR %s

// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-unknown-aix -mignore-xcoff-visibility -fvisibility default -emit-llvm -round-trip-args -o - -x c++ %s  | \
// RUN: FileCheck -check-prefix=NOVISIBILITY-IR %s

// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-unknown-aix -fvisibility default -emit-llvm -o - -x c++ %s  | \
// RUN: FileCheck -check-prefix=VISIBILITY-IR %s

// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-unknown-aix -fvisibility default -round-trip-args -emit-llvm -o - -x c++ %s  | \
// RUN: FileCheck -check-prefix=VISIBILITY-IR %s

__attribute__((visibility("hidden"))) void foo_h(int *p) {
  (*p)++;
}

__attribute__((visibility("protected"))) int b;

extern __attribute__((visibility("hidden"))) void zoo_extern_h(void);

void (*foo_p)(void) = zoo_extern_h;

__attribute__((visibility("protected"))) void bar() {
  foo_h(&b);
  foo_p();
}

class TestClass {
public:
  __attribute__((__visibility__("hidden"))) int value() const noexcept { return 0; }
};

int main() {
  TestClass TC;
  return TC.value();
}

template <class T>
class basic {
public:
  __attribute__((__visibility__("protected"))) int getdata() { return 1; }
};

template class basic<int>;

#pragma GCC visibility push(hidden)
int pramb;
void prambar() {}
#pragma GCC visibility pop

// VISIBILITY-IR:    @b = protected global i32 0
// VISIBILITY-IR:    @pramb = hidden global i32 0
// VISIBILITY-IR:    define hidden void @_Z5foo_hPi(i32* noundef %p)
// VISIBILITY-IR:    declare hidden void @_Z12zoo_extern_hv()
// VISIBILITY-IR:    define protected void @_Z3barv()
// VISIBILITY-IR:    define linkonce_odr hidden noundef i32 @_ZNK9TestClass5valueEv(%class.TestClass* {{[^,]*}} %this)
// VISIBILITY-IR:    define weak_odr protected noundef i32 @_ZN5basicIiE7getdataEv(%class.basic* {{[^,]*}} %this)
// VISIBILITY-IR:    define hidden void @_Z7prambarv()

// NOVISIBILITY-IR:    @b = global i32 0
// NOVISIBILITY-IR:    @pramb = global i32 0
// NOVISIBILITY-IR:    define void @_Z5foo_hPi(i32* noundef %p)
// NOVISIBILITY-IR:    declare void @_Z12zoo_extern_hv()
// NOVISIBILITY-IR:    define void @_Z3barv()
// NOVISIBILITY-IR:    define linkonce_odr noundef i32 @_ZNK9TestClass5valueEv(%class.TestClass* {{[^,]*}} %this)
// NOVISIBILITY-IR:    define weak_odr noundef i32 @_ZN5basicIiE7getdataEv(%class.basic* {{[^,]*}} %this)
// NOVISIBILITY-IR:    define void @_Z7prambarv()
