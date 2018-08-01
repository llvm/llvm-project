// RUN: rm -rf %t

// Set up directories
// RUN: mkdir -p %t/APINotes
// RUN: cp %S/Inputs/APINotes/SomeOtherKit.apinotes %t/APINotes/SomeOtherKit.apinotes
// RUN: mkdir -p %t/Frameworks
// RUN: cp -r %S/Inputs/Frameworks/SomeOtherKit.framework %t/Frameworks

// First build: check that 'methodB' is unavailable but 'methodA' is available.
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -Rmodule-build -fmodules-cache-path=%t/ModulesCache -iapinotes-modules %t/APINotes  -F %t/Frameworks %s > %t/before.log 2>&1
// RUN: FileCheck -check-prefix=CHECK-METHODB %s < %t/before.log
// RUN: FileCheck -check-prefix=CHECK-REBUILD %s < %t/before.log
// RUN: FileCheck -check-prefix=CHECK-ONE-ERROR %s < %t/before.log

// Do it again; now we're using caches.
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -Rmodule-build -fmodules-cache-path=%t/ModulesCache -iapinotes-modules %t/APINotes  -F %t/Frameworks %s > %t/before.log 2>&1
// RUN: FileCheck -check-prefix=CHECK-METHODB %s < %t/before.log
// RUN: FileCheck -check-prefix=CHECK-WITHOUT-REBUILD %s < %t/before.log
// RUN: FileCheck -check-prefix=CHECK-ONE-ERROR %s < %t/before.log

// Add a blank line to the header to force the module to rebuild, without
// (yet) changing API notes.
// RUN: echo >> %t/Frameworks/SomeOtherKit.framework/Headers/SomeOtherKit.h
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -Rmodule-build -fmodules-cache-path=%t/ModulesCache -iapinotes-modules %t/APINotes  -F %t/Frameworks %s > %t/before.log 2>&1
// RUN: FileCheck -check-prefix=CHECK-METHODB %s < %t/before.log
// RUN: FileCheck -check-prefix=CHECK-REBUILD %s < %t/before.log
// RUN: FileCheck -check-prefix=CHECK-ONE-ERROR %s < %t/before.log

// Change the API notes file, after the module has rebuilt once.
// RUN: echo '      - Selector: "methodA"' >> %t/APINotes/SomeOtherKit.apinotes
// RUN: echo '        MethodKind: Instance' >> %t/APINotes/SomeOtherKit.apinotes
// RUN: echo '        Availability: none' >> %t/APINotes/SomeOtherKit.apinotes
// RUN: echo '        AvailabilityMsg: "not here either"' >> %t/APINotes/SomeOtherKit.apinotes

// Build again: check that both methods are now unavailable and that the module rebuilt.
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -Rmodule-build -fmodules-cache-path=%t/ModulesCache -iapinotes-modules %t/APINotes  -F %t/Frameworks %s > %t/after.log 2>&1
// RUN: FileCheck -check-prefix=CHECK-METHODA %s < %t/after.log
// RUN: FileCheck -check-prefix=CHECK-METHODB %s < %t/after.log
// RUN: FileCheck -check-prefix=CHECK-REBUILD %s < %t/after.log
// RUN: FileCheck -check-prefix=CHECK-TWO-ERRORS %s < %t/after.log

// Run the build again: check that both methods are now unavailable
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -Rmodule-build -fmodules-cache-path=%t/ModulesCache -iapinotes-modules %t/APINotes  -F %t/Frameworks %s > %t/after.log 2>&1
// RUN: FileCheck -check-prefix=CHECK-METHODA %s < %t/after.log
// RUN: FileCheck -check-prefix=CHECK-METHODB %s < %t/after.log
// RUN: FileCheck -check-prefix=CHECK-WITHOUT-REBUILD %s < %t/after.log
// RUN: FileCheck -check-prefix=CHECK-TWO-ERRORS %s < %t/after.log

// Set up a directory with pre-compiled API notes.
// RUN: mkdir -p %t/CompiledAPINotes
// RUN: rm -rf %t/ModulesCache
// RUN: rm -rf %t/APINotesCache
// RUN: %clang -cc1apinotes -yaml-to-binary -o %t/CompiledAPINotes/SomeOtherKit.apinotesc %S/Inputs/APINotes/SomeOtherKit.apinotes

// First build: check that 'methodB' is unavailable but 'methodA' is available.
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -Rmodule-build -fmodules-cache-path=%t/ModulesCache -iapinotes-modules %t/CompiledAPINotes  -F %t/Frameworks %s > %t/compiled-before.log 2>&1
// RUN: FileCheck -check-prefix=CHECK-METHODB %s < %t/compiled-before.log
// RUN: FileCheck -check-prefix=CHECK-REBUILD %s < %t/compiled-before.log
// RUN: FileCheck -check-prefix=CHECK-ONE-ERROR %s < %t/compiled-before.log

// Do it again; now we're using caches.
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -Rmodule-build -fmodules-cache-path=%t/ModulesCache -iapinotes-modules %t/CompiledAPINotes  -F %t/Frameworks %s > %t/compiled-before.log 2>&1
// RUN: FileCheck -check-prefix=CHECK-METHODB %s < %t/compiled-before.log
// RUN: FileCheck -check-prefix=CHECK-WITHOUT-REBUILD %s < %t/compiled-before.log
// RUN: FileCheck -check-prefix=CHECK-ONE-ERROR %s < %t/compiled-before.log

// Compile a new API notes file to replace the old one.
// RUN: %clang -cc1apinotes -yaml-to-binary -o %t/CompiledAPINotes/SomeOtherKit.apinotesc %t/APINotes/SomeOtherKit.apinotes

// Build again: check that both methods are now unavailable and that the module rebuilt.
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -Rmodule-build -fmodules-cache-path=%t/ModulesCache -iapinotes-modules %t/CompiledAPINotes  -F %t/Frameworks %s > %t/compiled-after.log 2>&1
// RUN: FileCheck -check-prefix=CHECK-METHODA %s < %t/compiled-after.log
// RUN: FileCheck -check-prefix=CHECK-METHODB %s < %t/compiled-after.log
// RUN: FileCheck -check-prefix=CHECK-REBUILD %s < %t/compiled-after.log
// RUN: FileCheck -check-prefix=CHECK-TWO-ERRORS %s < %t/compiled-after.log

// Run the build again: check that both methods are now unavailable
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -Rmodule-build -fmodules-cache-path=%t/ModulesCache -iapinotes-modules %t/CompiledAPINotes  -F %t/Frameworks %s > %t/compiled-after.log 2>&1
// RUN: FileCheck -check-prefix=CHECK-METHODA %s < %t/compiled-after.log
// RUN: FileCheck -check-prefix=CHECK-METHODB %s < %t/compiled-after.log
// RUN: FileCheck -check-prefix=CHECK-WITHOUT-REBUILD %s < %t/compiled-after.log
// RUN: FileCheck -check-prefix=CHECK-TWO-ERRORS %s < %t/compiled-after.log

@import SomeOtherKit;

void test(A *a) {
  // CHECK-METHODA: error: 'methodA' is unavailable: not here either
  [a methodA];

  // CHECK-METHODB: error: 'methodB' is unavailable: anything but this
  [a methodB];
}

// CHECK-REBUILD: remark: building module{{.*}}SomeOtherKit

// CHECK-WITHOUT-REBUILD-NOT: remark: building module{{.*}}SomeOtherKit

// CHECK-ONE-ERROR: 1 error generated.
// CHECK-TWO-ERRORS: 2 errors generated.

