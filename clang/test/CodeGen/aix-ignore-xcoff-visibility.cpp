// RUN: %clang_cc1 -triple powerpc-unknown-aix -o - -x c++ -S  %s  |\
// RUN:   FileCheck --check-prefix=IGNOREVISIBILITY-ASM %s

// RUN: %clang_cc1 -triple powerpc-unknown-aix -mignore-xcoff-visibility -o - -x c++ -S %s  | \
// RUN: FileCheck -check-prefix=IGNOREVISIBILITY-ASM %s

// RUN: %clang_cc1 -triple powerpc-unknown-aix -mignore-xcoff-visibility -fvisibility default -o - -x c++ -S %s  | \
// RUN: FileCheck -check-prefix=IGNOREVISIBILITY-ASM %s

// RUN: %clang_cc1 -triple powerpc-unknown-aix -fvisibility default -o - -x c++ -S %s  | \
// RUN: FileCheck -check-prefix=VISIBILITY-ASM %s

// RUN: %clang_cc1 -triple powerpc-unknown-aix -mignore-xcoff-visibility -fvisibility default -o - -x c++ -S %s  | \
// RUN: FileCheck -check-prefix=IGNOREVISIBILITY-ASM %s

// RUN: %clang_cc1 -triple powerpc-unknown-aix -fvisibility default -o - -x c++ -S %s  | \
// RUN: FileCheck -check-prefix=VISIBILITY-ASM %s

// RUN: %clang_cc1 -triple powerpc-unknown-aix -mignore-xcoff-visibility -fvisibility default -emit-llvm -o - -x c++ %s  | \
// RUN: FileCheck -check-prefix=VISIBILITY-IR %s

// RUN: %clang_cc1 -triple powerpc-unknown-aix -fvisibility default -emit-llvm -o - -x c++ %s  | \
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
// VISIBILITY-IR:    define hidden void @_Z5foo_hPi(i32* %p)
// VISIBILITY-IR:    declare hidden void @_Z12zoo_extern_hv()
// VISIBILITY-IR:    define protected void @_Z3barv()
// VISIBILITY-IR:    define linkonce_odr hidden i32 @_ZNK9TestClass5valueEv(%class.TestClass* %this)
// VISIBILITY-IR:    define weak_odr protected i32 @_ZN5basicIiE7getdataEv(%class.basic* %this)
// VISIBILITY-IR:    define hidden void @_Z7prambarv()

// VISIBILITY-ASM: .globl  _Z5foo_hPi[DS],hidden
// VISIBILITY-ASM: .globl  ._Z5foo_hPi,hidden
// VISIBILITY-ASM: .globl  _Z3barv[DS],protected
// VISIBILITY-ASM: .globl  ._Z3barv,protected
// VISIBILITY-ASM: .weak   _ZNK9TestClass5valueEv[DS],hidden
// VISIBILITY-ASM: .weak   ._ZNK9TestClass5valueEv,hidden
// VISIBILITY-ASM: .weak   _ZN5basicIiE7getdataEv[DS],protected
// VISIBILITY-ASM: .weak   ._ZN5basicIiE7getdataEv,protected
// VISIBILITY-ASM: .globl  _Z7prambarv[DS],hidden
// VISIBILITY-ASM: .globl  ._Z7prambarv,hidden
// VISIBILITY-ASM: .globl  b,protected
// VISIBILITY-ASM: .globl  pramb,hidden

// IGNOREVISIBILITY-ASM: .globl  _Z5foo_hPi[DS]
// IGNOREVISIBILITY-ASM: .globl  ._Z5foo_hPi
// IGNOREVISIBILITY-ASM: .globl  _Z3barv[DS]
// IGNOREVISIBILITY-ASM: .globl  ._Z3barv
// IGNOREVISIBILITY-ASM: .weak   _ZNK9TestClass5valueEv[DS]
// IGNOREVISIBILITY-ASM: .weak   ._ZNK9TestClass5valueEv
// IGNOREVISIBILITY-ASM: .weak   _ZN5basicIiE7getdataEv[DS]
// IGNOREVISIBILITY-ASM: .weak   ._ZN5basicIiE7getdataEv
// IGNOREVISIBILITY-ASM: .globl  _Z7prambarv[DS]
// IGNOREVISIBILITY-ASM: .globl  ._Z7prambarv
// IGNOREVISIBILITY-ASM: .globl  b
// IGNOREVISIBILITY-ASM: .globl  pramb
