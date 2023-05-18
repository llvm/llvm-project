// Test creation of modules that include extension blocks.
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -ftest-module-file-extension=clang.testA:1:5:0:user_info_for_A -ftest-module-file-extension=clang.testB:2:3:0:user_info_for_B -fmodules-cache-path=%t -I %S/Inputs %s

// Make sure the extension blocks are actually there.
// RUN: llvm-bcanalyzer %t/ExtensionTestA.pcm | FileCheck -check-prefix=CHECK-BCANALYZER %s
// RUN: %clang_cc1 -module-file-info %t/ExtensionTestA.pcm | FileCheck -check-prefix=CHECK-INFO %s

// Make sure that the readers are able to check the metadata.
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -ftest-module-file-extension=clang.testA:1:5:0:user_info_for_A -ftest-module-file-extension=clang.testB:2:3:0:user_info_for_B -fmodules-cache-path=%t -I %S/Inputs %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -ftest-module-file-extension=clang.testA:1:3:0:user_info_for_A -ftest-module-file-extension=clang.testB:3:2:0:user_info_for_B -fmodules-cache-path=%t -I %S/Inputs %s -verify

// Make sure that extension blocks can be part of the module hash.
// We test this in an obscure way, by making sure we don't get conflicts when
// using different "versions" of the extensions. Above, the "-verify" test
// checks that such conflicts produce errors.
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -ftest-module-file-extension=clang.testA:1:5:1:user_info_for_A -ftest-module-file-extension=clang.testB:2:3:1:user_info_for_B -fmodules-cache-path=%t -I %S/Inputs %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -ftest-module-file-extension=clang.testA:1:3:1:user_info_for_A -ftest-module-file-extension=clang.testB:3:2:1:user_info_for_B -fmodules-cache-path=%t -I %S/Inputs %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -ftest-module-file-extension=clang.testA:2:5:0:user_info_for_A -ftest-module-file-extension=clang.testB:7:3:0:user_info_for_B -fmodules-cache-path=%t -I %S/Inputs %s

// Make sure we can read the message back.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -ftest-module-file-extension=clang.testA:1:5:0:user_info_for_A -ftest-module-file-extension=clang.testB:2:3:0:user_info_for_B -fmodules-cache-path=%t -I %S/Inputs %s > %t.log 2>&1
// RUN: FileCheck -check-prefix=CHECK-MESSAGE %s < %t.log

// Make sure we diagnose duplicate module file extensions.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -ftest-module-file-extension=clang.testA:1:5:0:user_info_for_A -ftest-module-file-extension=clang.testA:1:5:0:user_info_for_A -fmodules-cache-path=%t -I %S/Inputs %s > %t.log 2>&1
// RUN: FileCheck -check-prefix=CHECK-DUPLICATE %s < %t.log

#include "ExtensionTestA.h"
// expected-error@-1{{test module file extension 'clang.testA' has different version (1.5) than expected (1.3)}}
// expected-error@-2{{test module file extension 'clang.testB' has different version (2.3) than expected (3.2)}}

// CHECK-BCANALYZER: {{Block ID.*EXTENSION_BLOCK}}
// CHECK-BCANALYZER: {{100.00.*EXTENSION_METADATA}}

// CHECK-INFO: Module file extension 'clang.testA' 1.5: user_info_for_A
// CHECK-INFO: Module file extension 'clang.testB' 2.3: user_info_for_B

// CHECK-MESSAGE: Read extension block message: Hello from clang.testA v1.5
// CHECK-MESSAGE: Read extension block message: Hello from clang.testB v2.3

// CHECK-DUPLICATE: warning: duplicate module file extension block name 'clang.testA'
