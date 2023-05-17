int a [[gnu::used]];
// RUN: %clang_cc1 -code-completion-at=%s:%(line-1):9 %s | FileCheck --check-prefix=STD %s
// STD:     COMPLETION: Pattern : __carries_dependency__
// STD-NOT: COMPLETION: Pattern : __convergent__
// STD:     COMPLETION: Pattern : __gnu__::__used__
// STD-NOT: COMPLETION: Pattern : __gnu__::used
// STD-NOT: COMPLETION: Pattern : __used__
// STD:     COMPLETION: Pattern : _Clang::__convergent__
// STD:     COMPLETION: Pattern : carries_dependency
// STD-NOT: COMPLETION: Pattern : clang::called_once
// STD:     COMPLETION: Pattern : clang::convergent
// STD-NOT: COMPLETION: Pattern : convergent
// STD-NOT: COMPLETION: Pattern : gnu::__used__
// STD:     COMPLETION: Pattern : gnu::abi_tag(<#Tags...#>)
// STD:     COMPLETION: Pattern : gnu::alias(<#Aliasee#>)
// STD:     COMPLETION: Pattern : gnu::used
// STD-NOT: COMPLETION: Pattern : used
// RUN: %clang_cc1 -code-completion-at=%s:%(line-17):9 -xobjective-c++ %s | FileCheck --check-prefix=STD-OBJC %s
// STD-OBJC: COMPLETION: Pattern : clang::called_once
// RUN: %clang_cc1 -code-completion-at=%s:%(line-19):14 %s | FileCheck --check-prefix=STD-NS %s
// STD-NS-NOT: COMPLETION: Pattern : __used__
// STD-NS-NOT: COMPLETION: Pattern : carries_dependency
// STD-NS-NOT: COMPLETION: Pattern : clang::convergent
// STD-NS-NOT: COMPLETION: Pattern : convergent
// STD-NS-NOT: COMPLETION: Pattern : gnu::used
// STD-NS:     COMPLETION: Pattern : used
int b [[__gnu__::used]];
// RUN: %clang_cc1 -code-completion-at=%s:%(line-1):18 %s | FileCheck --check-prefix=STD-NSU %s
// STD-NSU:     COMPLETION: Pattern : __used__
// STD-NSU-NOT: COMPLETION: Pattern : used

int c [[using gnu: used]];
// RUN: %clang_cc1 -code-completion-at=%s:%(line-1):15 %s | FileCheck --check-prefix=STD-USING %s
// STD-USING:     COMPLETION: __gnu__
// STD-USING:     COMPLETION: _Clang
// STD-USING-NOT: COMPLETION: Pattern : carries_dependency
// STD-USING:     COMPLETION: clang
// STD-USING-NOT: COMPLETION: Pattern : clang::
// STD-USING-NOT: COMPLETION: Pattern : gnu::
// STD-USING:     COMPLETION: gnu
// RUN: %clang_cc1 -code-completion-at=%s:%(line-9):20 %s | FileCheck --check-prefix=STD-NS %s

int d __attribute__((used));
// RUN: %clang_cc1 -code-completion-at=%s:%(line-1):22 %s | FileCheck --check-prefix=GNU %s
// GNU:     COMPLETION: Pattern : __carries_dependency__
// GNU:     COMPLETION: Pattern : __convergent__
// GNU-NOT: COMPLETION: Pattern : __gnu__::__used__
// GNU:     COMPLETION: Pattern : __used__
// GNU-NOT: COMPLETION: Pattern : _Clang::__convergent__
// GNU:     COMPLETION: Pattern : carries_dependency
// GNU-NOT: COMPLETION: Pattern : clang::convergent
// GNU:     COMPLETION: Pattern : convergent
// GNU-NOT: COMPLETION: Pattern : gnu::used
// GNU:     COMPLETION: Pattern : used

#pragma clang attribute push (__attribute__((internal_linkage)), apply_to=variable)
int e;
#pragma clang attribute pop
// RUN: %clang_cc1 -code-completion-at=%s:%(line-3):46 %s | FileCheck --check-prefix=PRAGMA %s
// PRAGMA: COMPLETION: Pattern : internal_linkage

#ifdef MS_EXT
int __declspec(thread) f;
// RUN: %clang_cc1 -fms-extensions -DMS_EXT -code-completion-at=%s:%(line-1):16 %s | FileCheck --check-prefix=DS %s
// DS-NOT: COMPLETION: Pattern : __convergent__
// DS-NOT: COMPLETION: Pattern : __used__
// DS-NOT: COMPLETION: Pattern : clang::convergent
// DS-NOT: COMPLETION: Pattern : convergent
// DS:     COMPLETION: Pattern : thread
// DS-NOT: COMPLETION: Pattern : used
// DS:     COMPLETION: Pattern : uuid

[uuid("123e4567-e89b-12d3-a456-426614174000")] struct g;
// RUN: %clang_cc1 -fms-extensions -DMS_EXT -code-completion-at=%s:%(line-1):2 %s | FileCheck --check-prefix=MS %s
// MS-NOT: COMPLETION: Pattern : __uuid__
// MS-NOT: COMPLETION: Pattern : clang::convergent
// MS-NOT: COMPLETION: Pattern : convergent
// MS-NOT: COMPLETION: Pattern : thread
// MS-NOT: COMPLETION: Pattern : used
// MS:     COMPLETION: Pattern : uuid
#endif // MS_EXT

void foo() {
  [[omp::sequence(directive(parallel), directive(critical))]]
  {}
}
// FIXME: support for omp attributes would be nice.
// RUN: %clang_cc1 -fopenmp -code-completion-at=%s:%(line-4):5 %s | FileCheck --check-prefix=OMP-NS --allow-empty %s
// OMP-NS-NOT: COMPLETION: omp
// RUN: %clang_cc1 -fopenmp -code-completion-at=%s:%(line-6):10 %s | FileCheck --check-prefix=OMP-ATTR --allow-empty %s
// OMP-ATTR-NOT: COMPLETION: Pattern : sequence
// RUN: %clang_cc1 -fopenmp -code-completion-at=%s:%(line-8):19 %s | FileCheck --check-prefix=OMP-NESTED --allow-empty %s
// OMP-NESTED-NOT: COMPLETION: Pattern : directive
