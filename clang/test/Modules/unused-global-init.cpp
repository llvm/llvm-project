// RUN: rm -rf %t
//
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x c++ -I %S/Inputs/unused-global-init -triple %itanium_abi_triple -emit-module %S/Inputs/unused-global-init/module.modulemap -fmodule-name=init -o %t/init.pcm
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x c++ -I %S/Inputs/unused-global-init -triple %itanium_abi_triple -emit-module %S/Inputs/unused-global-init/module.modulemap -fmodule-name=unused -o %t/unused.pcm -fmodule-file=%t/init.pcm
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x c++ -I %S/Inputs/unused-global-init -triple %itanium_abi_triple -emit-module %S/Inputs/unused-global-init/module.modulemap -fmodule-name=used -o %t/used.pcm -fmodule-file=%t/init.pcm
//
// No module file: init.h performs init.
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x c++ -I %S/Inputs/unused-global-init -triple %itanium_abi_triple -emit-llvm -o - %s -DINIT | FileCheck --check-prefix=CHECK-INIT %s
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x c++ -I %S/Inputs/unused-global-init -triple %itanium_abi_triple -emit-llvm -o - %s -DUSED | FileCheck --check-prefix=CHECK-INIT %s
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x c++ -I %S/Inputs/unused-global-init -triple %itanium_abi_triple -emit-llvm -o - %s -DOTHER -DUNUSED | FileCheck --check-prefix=CHECK-NO-INIT %s
//
// With module files: if there is a transitive import of any part of the
// module, we run its global initializers (even if the imported piece is not
// visible here).
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x c++ -I %S/Inputs/unused-global-init -triple %itanium_abi_triple -emit-llvm -o - %s -fmodule-file=%t/used.pcm -fmodule-file=%t/unused.pcm -DINIT | FileCheck --check-prefix=CHECK-INIT %s
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x c++ -I %S/Inputs/unused-global-init -triple %itanium_abi_triple -emit-llvm -o - %s -fmodule-file=%t/used.pcm -fmodule-file=%t/unused.pcm -DOTHER | FileCheck --check-prefix=CHECK-NO-INIT %s
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x c++ -I %S/Inputs/unused-global-init -triple %itanium_abi_triple -emit-llvm -o - %s -fmodule-file=%t/used.pcm -fmodule-file=%t/unused.pcm -DUSED | FileCheck --check-prefix=CHECK-INIT %s
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x c++ -I %S/Inputs/unused-global-init -triple %itanium_abi_triple -emit-llvm -o - %s -fmodule-file=%t/used.pcm -fmodule-file=%t/unused.pcm -DUNUSED | FileCheck --check-prefix=CHECK-NO-INIT %s

#ifdef INIT
#include "init.h"
#endif

#ifdef OTHER
#include "other.h"
#endif

#ifdef USED
#include "used.h"
#endif

#ifdef UNUSED
#include "unused.h"
#endif

// CHECK-INIT: call {{.*}}@_ZN4InitC
// CHECK-NO-INIT-NOT: call {{.*}}@_ZN4InitC
