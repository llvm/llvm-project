int a [[gnu::used]];
// RUN: %clang_cc1 -code-completion-at=%s:1:9 %s | FileCheck --check-prefix=STD %s
// STD:     COMPLETION: __carries_dependency__
// STD-NOT: COMPLETION: __convergent__
// STD:     COMPLETION: __gnu__::__used__
// STD-NOT: COMPLETION: __gnu__::used
// STD-NOT: COMPLETION: __used__
// STD:     COMPLETION: _Clang::__convergent__
// STD:     COMPLETION: carries_dependency
// STD:     COMPLETION: clang::convergent
// STD-NOT: COMPLETION: convergent
// STD-NOT:     COMPLETION: gnu::__used__
// STD:     COMPLETION: gnu::used
// STD-NOT: COMPLETION: used
// RUN: %clang_cc1 -code-completion-at=%s:1:14 %s | FileCheck --check-prefix=STD-NS %s
// STD-NS-NOT: COMPLETION: __used__
// STD-NS-NOT: COMPLETION: carries_dependency
// STD-NS-NOT: COMPLETION: clang::convergent
// STD-NS-NOT: COMPLETION: convergent
// STD-NS-NOT: COMPLETION: gnu::used
// STD-NS:     COMPLETION: used
int b [[__gnu__::used]];
// RUN: %clang_cc1 -code-completion-at=%s:22:18 %s | FileCheck --check-prefix=STD-NSU %s
// STD-NSU:     COMPLETION: __used__
// STD-NSU-NOT: COMPLETION: used

int c [[using gnu: used]];
// RUN: %clang_cc1 -code-completion-at=%s:27:15 %s | FileCheck --check-prefix=STD-USING %s
// STD-USING:     COMPLETION: __gnu__
// STD-USING:     COMPLETION: _Clang
// STD-USING-NOT: COMPLETION: carries_dependency
// STD-USING:     COMPLETION: clang
// STD-USING-NOT: COMPLETION: clang::
// STD-USING-NOT: COMPLETION: gnu::
// STD-USING:     COMPLETION: gnu
// RUN: %clang_cc1 -code-completion-at=%s:27:20 %s | FileCheck --check-prefix=STD-NS %s

int d __attribute__((used));
// RUN: %clang_cc1 -code-completion-at=%s:38:22 %s | FileCheck --check-prefix=GNU %s
// GNU:     COMPLETION: __carries_dependency__
// GNU:     COMPLETION: __convergent__
// GNU-NOT: COMPLETION: __gnu__::__used__
// GNU:     COMPLETION: __used__
// GNU-NOT: COMPLETION: _Clang::__convergent__
// GNU:     COMPLETION: carries_dependency
// GNU-NOT: COMPLETION: clang::convergent
// GNU:     COMPLETION: convergent
// GNU-NOT: COMPLETION: gnu::used
// GNU:     COMPLETION: used

#pragma clang attribute push (__attribute__((internal_linkage)), apply_to=variable)
int e;
#pragma clang attribute pop
// RUN: %clang_cc1 -code-completion-at=%s:51:46 %s | FileCheck --check-prefix=PRAGMA %s
// PRAGMA: internal_linkage

#ifdef MS_EXT
int __declspec(thread) f;
// RUN: %clang_cc1 -fms-extensions -DMS_EXT -code-completion-at=%s:58:16 %s | FileCheck --check-prefix=DS %s
// DS-NOT: COMPLETION: __convergent__
// DS-NOT: COMPLETION: __used__
// DS-NOT: COMPLETION: clang::convergent
// DS-NOT: COMPLETION: convergent
// DS:     COMPLETION: thread
// DS-NOT: COMPLETION: used
// DS:     COMPLETION: uuid

[uuid("123e4567-e89b-12d3-a456-426614174000")] struct g;
// RUN: %clang_cc1 -fms-extensions -DMS_EXT -code-completion-at=%s:68:2 %s | FileCheck --check-prefix=MS %s
// MS-NOT: COMPLETION: __uuid__
// MS-NOT: COMPLETION: clang::convergent
// MS-NOT: COMPLETION: convergent
// MS-NOT: COMPLETION: thread
// MS-NOT: COMPLETION: used
// MS:     COMPLETION: uuid
#endif // MS_EXT

void foo() {
  [[omp::sequence(directive(parallel), directive(critical))]]
  {}
}
// FIXME: support for omp attributes would be nice.
// RUN: %clang_cc1 -fopenmp -code-completion-at=%s:79:5 %s | FileCheck --check-prefix=OMP-NS --allow-empty %s
// OMP-NS-NOT: omp
// RUN: %clang_cc1 -fopenmp -code-completion-at=%s:79:10 %s | FileCheck --check-prefix=OMP-ATTR --allow-empty %s
// OMP-ATTR-NOT: sequence
// RUN: %clang_cc1 -fopenmp -code-completion-at=%s:79:19 %s | FileCheck --check-prefix=OMP-NESTED --allow-empty %s
// OMP-NESTED-NOT: directive
