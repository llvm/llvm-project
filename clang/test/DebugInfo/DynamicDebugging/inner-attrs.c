// RUN: %clang_cc1 -triple %itanium_abi_triple %s -Os -debug-info-kind=constructor -fdynamic-debugging -o %t \
// RUN:    -emit-llvm --save-dynamic-debugging-temps --discard-dynamic-debugging-debug-module
// RUN: FileCheck %s --check-prefix=INPUT < %t.dyndbg.0.input.ll
// RUN: FileCheck %s --check-prefix=INNER < %t.dyndbg.1.inner.ll

/// always_inline should be removed from the inner module.
__attribute__((always_inline)) void a() { }
__attribute__((minsize)) void b() { }

/// Confirm the input module has alwaysinline, minsize, optsize.
// INPUT: define dso_local void @a() #0
// INPUT: define dso_local void @b() #1
// INPUT: attributes #0 = { alwaysinline nounwind optsize "
// INPUT: attributes #1 = { minsize nounwind optsize "

/// Check the inner module has noinline and optnone added to its copies of the
/// outer functions, removing alwaysinline, minsize, optsize.
// INNER: define dso_local void @__dyndbg.a() #0
// INNER: define dso_local void @__dyndbg.b() #0
// INNER: declare dso_local void @a() #1
// INNER: declare dso_local void @b() #2
// INNER: attributes #0 = { noinline nounwind optnone "
/// Inner's references to outer's functions keep their original attributes.
// INNER: attributes #1 = { alwaysinline nounwind optsize "
// INNER: attributes #2 = { minsize nounwind optsize "
