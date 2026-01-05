// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// This test builds two PCHs. bridging.h.pch depends on h1.h.pch.
// Then the test uses bridiging.h.pch in a source file that imports
// a module with config macros.
// This is a normal use case and no warnings should be issued.
// RUN: %clang_cc1 -fmodules \
// RUN:   -fmodule-map-file=%S/Inputs/pch-config-macros/include/module.modulemap \
// RUN:   -fmodules-cache-path=%t/cache -I %S/Inputs/pch-config-macros/include \
// RUN:   h1.h -emit-pch -o h1.h.pch -DCONFIG1 -DCONFIG2
// RUN: %clang_cc1 -fmodules \
// RUN:   -fmodule-map-file=%S/Inputs/pch-config-macros/include/module.modulemap \
// RUN:   -fmodules-cache-path=%t/cache -I %S/Inputs/pch-config-macros/include \
// RUN:   -include-pch %t/h1.h.pch bridging.h -emit-pch -o bridging.h.pch \
// RUN:   -DCONFIG1 -DCONFIG2
// RUN: %clang_cc1 -fmodules \
// RUN:   -fmodule-map-file=%S/Inputs/pch-config-macros/include/module.modulemap \
// RUN:   -fmodules-cache-path=%t/cache -I %S/Inputs/pch-config-macros/include \
// RUN:   -emit-obj -o main.o main.c -include-pch %t/bridging.h.pch \
// RUN:   -DCONFIG1 -DCONFIG2 -verify

//--- h1.h
#if CONFIG1
int bar1() { return 42; }
#else
int bar2() { return 43; }
#endif

//--- bridging.h
#if CONFIG1
int bar() { return bar1(); }
#else
int bar() { return bar2(); }
#endif

#if CONFIG2
int baz() { return 77; }
#endif

//--- main.c
#include "Mod1.h"
// expected-no-diagnostics

int main_func() {
    return foo() + bar(); 
}
