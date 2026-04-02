// RUN: %clang -cc1 -triple %itanium_abi_triple %s -debug-info-kind=limited -fdynamic-debugging -o %t \
// RUN:    -emit-llvm --save-dynamic-debugging-temps --discard-dynamic-debugging-debug-module
// RUN: FileCheck %s --check-prefix=INNER < %t.dyndbg.1.inner.ll
// RUN: FileCheck %s --check-prefix=OUTER < %t.dyndbg.2.outer.ll

/// Test functions get expected linkage and names in the dyndbg inner
/// and outer modules. Each outer (to-be-optimized) function should have a
/// corresponding inner (unoptimized) version.
///
/// The internal functions get external aliases in outer.

// OUTER: $_Z7odrweakv = comdat any
// INNER: $__dyndbg._Z7odrweakv = comdat any

/// Outer: as input. Inner: external reference, __dyndbg copy.
void external() {}
// OUTER-DAG: define  dso_local void @_Z8externalv()
// INNER-DAG: declare dso_local void @_Z8externalv()
// INNER-DAG: define  dso_local void @__dyndbg._Z8externalv()

/// Outer: add external linkage alias. Inner: external reference to alias,
/// __dyndbg copy.
[[gnu::used]] static void internal() {}
// OUTER-DAG: @_ZL8internalv.dyndbg.[[hash:[0-9A-Z]+]] = hidden alias void (), ptr @_ZL8internalv
// OUTER-DAG: define  internal void @_ZL8internalv()
// INNER-DAG: declare hidden void @_ZL8internalv.dyndbg.[[hash:[0-9A-Z]+]]()
// INNER-DAG: define  hidden void @__dyndbg._ZL8internalv.dyndbg.[[hash]]

/// Outer: as input. Inner: external reference, __dyndbg copy.
[[gnu::used]] inline void odrweak() {}
// OUTER-DAG: define  linkonce_odr void @_Z7odrweakv()
// INNER-DAG: declare void @_Z7odrweakv()
// INNER-DAG: define  linkonce_odr void @__dyndbg._Z7odrweakv()
