// RUN: %clang -cc1 -triple %itanium_abi_triple %s -debug-info-kind=limited -fdynamic-debugging -o %t \
// RUN:    -emit-llvm --save-dynamic-debugging-temps --discard-dynamic-debugging-debug-module
// RUN: FileCheck %s --check-prefix=INNER < %t.dyndbg.1.inner.ll --implicit-check-not=@__dyndbg._ZN1X5weirdE
// RUN: FileCheck %s --check-prefix=OUTER < %t.dyndbg.2.outer.ll

/// Check that internal symbols in comdats, __cxx_global_var_init in this case,
/// are not promoted (no external alias is produced).
///
/// --implicit-check-not=@__dyndbg._ZN1X5weirdE:
/// The innner cxx_global_var_init gets a comdat, '$__dyndbg._ZN1X5weirdE', but
/// unlike the one in the outer module, this one has no associated global data.

// OUTER: @_ZN1X5weirdE = linkonce_odr global %struct.X zeroinitializer, comdat, align 4
// INNER: @_ZN1X5weirdE = external global %struct.X, align 4

// OUTER: define  internal void @__cxx_global_var_init() {{.*}} comdat($_ZN1X5weirdE)
// INNER: define  internal void @__dyndbg.__cxx_global_var_init() {{.*}} comdat($__dyndbg._ZN1X5weirdE)
// INNER: declare dso_local void @__cxx_global_var_init()

struct X {
public:
  X(int a) : a(a) {}
  int a;
  static const X weird;
};

inline const X X::weird = X(5);
