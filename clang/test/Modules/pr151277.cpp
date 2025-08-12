// RUN: split-file %s %t

// RUN: %clang_cc1 -I%t -emit-module -o %t/a.pcm -fmodules %t/module.modulemap -fno-implicit-modules -fmodule-name=a -x c++-header -fms-compatibility
// RUN: %clang_cc1 -I%t -emit-module -o %t/b.pcm -fmodules %t/module.modulemap -fno-implicit-modules -fmodule-name=b -x c++-header -fms-compatibility -fmodule-file=%t/a.pcm

//--- module.modulemap
module a { header "a.h" }
module b { header "b.h" }

//--- a.h
type_info* foo;

//--- b.h
type_info* bar;

