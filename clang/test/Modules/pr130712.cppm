// RUN: split-file %s %t

// There are two requirements here to result in the owner of a macro being null.
// 1) There must be a configuration mismatch between a header and a file it depends on
// 2) -fmodules-local-submodule-visibility must be enabled.

// In the following example, when compiling module C, A_H has no owning module.

// RUN: %clang_cc1 -I%t -emit-module -o %t/a.pcm -fmodules %t/module.modulemap -fmodule-name=a -fmodules-local-submodule-visibility 
// RUN: %clang_cc1 -fexceptions -Wno-module-file-config-mismatch -I%t -emit-module -o %t/b.pcm -fmodules %t/module.modulemap -fmodule-name=b -fmodules-local-submodule-visibility -fmodule-file=%t/a.pcm
// RUN: %clang_cc1 -fexceptions -Wno-module-file-config-mismatch -I%t -emit-module -o %t/c.pcm -fmodules %t/module.modulemap -fmodule-name=c -fmodules-local-submodule-visibility -fmodule-file=%t/a.pcm -fmodule-file=%t/b.pcm

//--- module.modulemap
module a { header "a.h" }
module b { header "b.h" }
module c { header "c.h" }

//--- a.h
#ifndef A_H
#define A_H
#endif

//--- b.h
#ifndef B_H
#define B_H

#include <a.h>

#endif

//--- c.h
#include <a.h>
#include <b.h>
