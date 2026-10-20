// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache -I %t/include %t/test.m
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache -I %t/include %t/test.m -fmodulemap-allow-subdirectory-search
// RUN: not %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache -I %t/include %t/test.m -fno-modulemap-allow-subdirectory-search

//--- include/UnrelatedName/Header.h
// empty

//--- include/UnrelatedName/module.modulemap
module UsefulCode {
  header "Header.h"
  export *
}

//--- test.m
@import UsefulCode;
