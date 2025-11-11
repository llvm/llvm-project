// Requires a specific target as the "module file has a different size than expected" is not reliable on all architectures.
// REQUIRES: x86-registered-target

// RUN: rm -rf %t

// -------------------------------
// Build chained modules A, B, and C
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-name=a -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/a.pcm \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty
//
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-name=b -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/b.pcm \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty
//
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/b.pcm \
// RUN:            -fmodule-name=c -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/c.pcm \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty
//
// CHECK-NO-IMPLICIT-BUILD-NOT: building module

// -------------------------------
// Build B with an implicit build of A
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-name=b -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/b-not-a.pcm \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-B-NO-A %s
//
// CHECK-B-NO-A: While building module 'b':
// CHECK-B-NO-A: building module 'a' as

// -------------------------------
// Check that we can use the explicitly-built A, B, and C modules.
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -verify %s -DHAVE_A
//
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-map-file=%S/Inputs/explicit-build/module.modulemap \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -verify %s -DHAVE_A
//
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/b.pcm \
// RUN:            -verify %s -DHAVE_A -DHAVE_B
//
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=%t/b.pcm \
// RUN:            -verify %s -DHAVE_A -DHAVE_B
//
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=%t/b.pcm \
// RUN:            -fmodule-file=%t/c.pcm \
// RUN:            -verify %s -DHAVE_A -DHAVE_B -DHAVE_C
//
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=%t/c.pcm \
// RUN:            -verify %s -DHAVE_A -DHAVE_B -DHAVE_C

// -------------------------------
// Check that -fmodule-file= in a module build makes the file transitively
// available even if it's not used.
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fno-implicit-modules -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/b.pcm \
// RUN:            -fmodule-name=d -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/d.pcm \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty
//
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fno-implicit-modules -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/d.pcm \
// RUN:            -verify %s -DHAVE_A -DHAVE_B

#if HAVE_A
  #include "a.h"
  static_assert(a == 1, "");
#else
  const int use_a = a; // expected-error {{undeclared identifier}}
#endif

#if HAVE_B
  #include "b.h"
  static_assert(b == 2, "");
#else
  const int use_b = b; // expected-error {{undeclared identifier}}
#endif

#if HAVE_C
  #include "c.h"
  static_assert(c == 3, "");
#else
  const int use_c = c; // expected-error {{undeclared identifier}}
#endif

#if HAVE_A && HAVE_B && HAVE_C
// expected-no-diagnostics
#endif

// -------------------------------
// Check that we can use a mixture of implicit and explicit modules.
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/b-not-a.pcm \
// RUN:            -verify %s -DHAVE_A -DHAVE_B

// -------------------------------
// Try to use two different flavors of the 'a' module.
// RUN: not %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=%t/b-not-a.pcm \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-MULTIPLE-AS %s
//
// RUN: not %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=%t/b-not-a.pcm \
// RUN:            -fmodule-map-file=%S/Inputs/explicit-build/module.modulemap \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-MULTIPLE-AS %s
//
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-name=a -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/a-alt.pcm \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty
//
// RUN: not %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=%t/a-alt.pcm \
// RUN:            -fmodule-map-file=%S/Inputs/explicit-build/module.modulemap \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-MULTIPLE-AS %s
//
// RUN: not %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a-alt.pcm \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-map-file=%S/Inputs/explicit-build/module.modulemap \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-MULTIPLE-AS %s
//
// CHECK-MULTIPLE-AS: error: module 'a' is defined in both '{{.*[/\\]}}a{{.*}}.pcm' and '{{.*[/\\]}}a{{.*}}.pcm'

// -------------------------------
// Try to import a PCH with -fmodule-file=
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-name=a -emit-pch %S/Inputs/explicit-build/a.h -o %t/a.pch -DBUILDING_A_PCH \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty
//
// RUN: not %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pch \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-A-AS-PCH %s
//
// CHECK-A-AS-PCH: fatal error: precompiled file '{{.*}}a.pch' was not built as a module

// -------------------------------
// Try to import a non-AST file with -fmodule-file=
//
// RUN: touch %t/not.pcm
//
// RUN: not %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/not.pcm \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-BAD-FILE %s
//
// CHECK-BAD-FILE: fatal error: file '{{.*}}not.pcm' is not a valid module file: file too small to contain precompiled file magic

// RUN: not %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/nonexistent.pcm \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-NO-FILE %s
//
// CHECK-NO-FILE: fatal error: module file '{{.*}}nonexistent.pcm' not found: module file not found

// RUN: mv %t/a.pcm %t/a-tmp.pcm
// RUN: not %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/c.pcm \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-NO-FILE-INDIRECT %s
// RUN: mv %t/a-tmp.pcm %t/a.pcm
//
// CHECK-NO-FILE-INDIRECT:      error: module file '{{.*}}a.pcm' not found
// CHECK-NO-FILE-INDIRECT-NEXT: note: imported by module 'b' in '{{.*}}b.pcm'
// CHECK-NO-FILE-INDIRECT-NEXT: note: imported by module 'c' in '{{.*}}c.pcm'
// CHECK-NO-FILE-INDIRECT-NOT:  note:

// -------------------------------
// Check that we diagnose stale dependencies correctly when modules change.
//
// Trigger a rebuild of A with a different configuration (-DA_EXTRA_DEFINE) to make B and C out of date
// RUN: mv %t/a.pcm %t/a-tmp.pcm
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-name=a -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/a.pcm \
// RUN:            -DA_EXTRA_DEFINE \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty

// Try to use C, which depends on B, which depends on the now-changed A.
// RUN: not %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/c.pcm \
// RUN:            %s -DHAVE_A -DHAVE_B -DHAVE_C 2>&1 | FileCheck --check-prefix=CHECK-OUT-OF-DATE-INDIRECT %s
//
// CHECK-OUT-OF-DATE-INDIRECT: fatal error: module file '{{.*}}b.pcm' is out of date because dependency '{{.*}}a.pcm' has changed
// CHECK-OUT-OF-DATE-INDIRECT-NEXT: note: imported by module 'b' in '{{.*}}b.pcm'
// CHECK-OUT-OF-DATE-INDIRECT-NEXT: note: imported by module 'c' in '{{.*}}c.pcm'

// Rebuild B with the new A, leaving C out of date.
// RUN: mv %t/b.pcm %t/b-tmp.pcm
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-name=b -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/b.pcm \
// RUN:            -DA_EXTRA_DEFINE \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty
//
// Now only C is out of date. Try to use C.
// RUN: not %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/c.pcm \
// RUN:            %s -DHAVE_A -DHAVE_B -DHAVE_C 2>&1 | FileCheck --check-prefix=CHECK-OUT-OF-DATE-DIRECT %s
//
// CHECK-OUT-OF-DATE-DIRECT: fatal error: module file '{{.*}}c.pcm' is out of date because dependency '{{.*}}b.pcm' has changed
// CHECK-OUT-OF-DATE-DIRECT-NOT: fatal error: module file '{{.*}}b.pcm' is out of date
//
// RUN: mv %t/a-tmp.pcm %t/a.pcm
// RUN: mv %t/b-tmp.pcm %t/b.pcm

// -------------------------------
// Check that we don't get upset if B's timestamp is newer than C's.
// RUN: touch %t/b.pcm
//
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/c.pcm \
// RUN:            -verify %s -DHAVE_A -DHAVE_B -DHAVE_C
//
// ... but that we do get upset if our B is different from the B that C expects.
//
// RUN: cp %t/b-not-a.pcm %t/b.pcm
//
// RUN: not %clang_cc1 -triple=x86_64-linux-gnu -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/c.pcm \
// RUN:            %s -DHAVE_A -DHAVE_B -DHAVE_C 2>&1 | FileCheck --check-prefix=CHECK-MISMATCHED-B %s
//
// CHECK-MISMATCHED-B:      fatal error: module file '{{.*}}c.pcm' is out of date because dependency '{{.*}}b.pcm' has changed: module file has a different size than expected
// CHECK-MISMATCHED-B-NEXT: note: imported by module 'c'
// CHECK-MISMATCHED-B-NOT:  note:
