// RUN: rm -rf %t
// RUN: cd %S
// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -fmodules -x c++ -fmodules-cache-path=%t \
// RUN:   -iquote Inputs/merge-target-features \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -emit-module -fmodule-name=foo -o %t/foo.pcm \
// RUN:   -triple i386-unknown-unknown \
// RUN:   -target-cpu i386 -target-feature +sse2 \
// RUN:   Inputs/merge-target-features/module.modulemap
//
// RUN: not %clang_cc1 -fmodules -x c++ -fmodules-cache-path=%t \
// RUN:   -iquote Inputs/merge-target-features \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -fmodule-map-file=Inputs/merge-target-features/module.modulemap \
// RUN:   -fmodule-file=%t/foo.pcm \
// RUN:   -triple i386-unknown-unknown \
// RUN:   -target-cpu i386 \
// RUN:   -fsyntax-only merge-target-features.cpp 2>&1 \
// RUN:   | FileCheck --check-prefix=SUBSET --implicit-check-not=error: %s
// SUBSET: error: AST file '{{.*}}foo.pcm' was compiled with the target feature '+sse2' but the current translation unit is not
// SUBSET: error: {{.*}} configuration mismatch
//
// RUN: %clang_cc1 -fmodules -x c++ -fmodules-cache-path=%t \
// RUN:   -iquote Inputs/merge-target-features \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -fmodule-map-file=Inputs/merge-target-features/module.modulemap \
// RUN:   -fmodule-file=%t/foo.pcm \
// RUN:   -triple i386-unknown-unknown \
// RUN:   -target-cpu i386 -target-feature +sse2 \
// RUN:   -fsyntax-only merge-target-features.cpp 2>&1 \
// RUN:   | FileCheck --allow-empty --check-prefix=SAME %s
// SAME-NOT: error:
//
// RUN: %clang_cc1 -fmodules -x c++ -fmodules-cache-path=%t \
// RUN:   -iquote Inputs/merge-target-features \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -fmodule-map-file=Inputs/merge-target-features/module.modulemap \
// RUN:   -fmodule-file=%t/foo.pcm \
// RUN:   -triple i386-unknown-unknown \
// RUN:   -target-cpu i386 -target-feature +sse2 -target-feature +sse3 \
// RUN:   -fsyntax-only merge-target-features.cpp 2>&1 \
// RUN:   | FileCheck --allow-empty --check-prefix=SUPERSET %s
// SUPERSET-NOT: error:
//
// RUN: not %clang_cc1 -fmodules -x c++ -fmodules-cache-path=%t \
// RUN:   -iquote Inputs/merge-target-features \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -fmodule-map-file=Inputs/merge-target-features/module.modulemap \
// RUN:   -fmodule-file=%t/foo.pcm \
// RUN:   -triple i386-unknown-unknown \
// RUN:   -target-cpu i386 -target-feature +cx16 \
// RUN:   -fsyntax-only merge-target-features.cpp 2>&1 \
// RUN:   | FileCheck --check-prefix=MISMATCH --implicit-check-not=error: %s
// MISMATCH: error: AST file '{{.*}}foo.pcm' was compiled with the target feature '+sse2' but the current translation unit is not
// MISMATCH: error: current translation unit is compiled with the target feature '+cx16' but the AST file '{{.*}}foo.pcm' was not
// MISMATCH: error: {{.*}} configuration mismatch

#include "foo.h"

int test(int x) {
  return foo(x);
}
