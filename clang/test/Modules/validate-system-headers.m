// RUN: rm -rf %t/ModuleCache
// RUN: mkdir -p %t/Inputs/usr/include
// RUN: touch %t/Inputs/usr/include/foo.h
// RUN: echo 'module Foo [system] { header "foo.h" }' > %t/Inputs/usr/include/module.map

////
// Build a module using a system header
// RUN: %clang_cc1 -isystem %t/Inputs/usr/include -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModuleCache -fdisable-module-hash -x objective-c-header -fsyntax-only %s
// RUN: cp %t/ModuleCache/Foo.pcm %t/Foo.pcm.saved

////
// Modify the system header, and confirm that we don't notice without -fmodules-validate-system-headers.
// The pcm file in the cache should fail to validate.
// RUN: echo ' ' >> %t/Inputs/usr/include/foo.h
// RUN: %clang_cc1 -isystem %t/Inputs/usr/include -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModuleCache -fdisable-module-hash -x objective-c-header -fsyntax-only %s
// RUN: diff %t/ModuleCache/Foo.pcm %t/Foo.pcm.saved

////
// Now make sure we rebuild the module when -fmodules-validate-system-headers is set.
// RUN: %clang_cc1 -isystem %t/Inputs/usr/include -fmodules -fimplicit-module-maps -fmodules-validate-system-headers -fmodules-cache-path=%t/ModuleCache -fdisable-module-hash -x objective-c-header -fsyntax-only %s
// RUN: not diff %t/ModuleCache/Foo.pcm %t/Foo.pcm.saved


////
// This should override -fmodules-validate-once-per-build-session
// RUN: cp %t/ModuleCache/Foo.pcm %t/Foo.pcm.saved
// RUN: %clang_cc1 -isystem %t/Inputs/usr/include -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModuleCache -fdisable-module-hash -x objective-c-header -fsyntax-only %s -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session
// RUN: diff %t/ModuleCache/Foo.pcm %t/Foo.pcm.saved

// Modify the system header...
// RUN: echo ' ' >> %t/Inputs/usr/include/foo.h

// Don't recompile due to -fmodules-validate-once-per-build-session
// RUN: %clang_cc1 -isystem %t/Inputs/usr/include -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModuleCache -fdisable-module-hash -x objective-c-header -fsyntax-only %s -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session
// RUN: diff %t/ModuleCache/Foo.pcm %t/Foo.pcm.saved

// Now add -fmodules-validate-system-headers and rebuild. No recompilation due to -fmodules-validate-once-per-build-session
// RUN: %clang_cc1 -isystem %t/Inputs/usr/include -fmodules -fimplicit-module-maps -fmodules-validate-system-headers -fmodules-cache-path=%t/ModuleCache -fdisable-module-hash -x objective-c-header -fsyntax-only %s -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session
// RUN: diff %t/ModuleCache/Foo.pcm %t/Foo.pcm.saved

@import Foo;
