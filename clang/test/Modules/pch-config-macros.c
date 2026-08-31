// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// This test builds two PCHs. bridging.h.pch depends on h1.h.pch.
// Then the test uses bridiging.h.pch in a source file that imports
// a module with config macros.
// The warnings should not fire if the config macros are specified when
// building the pch and the main TU.
// This is a normal use case and no warnings should be issued.
// RUN: %clang_cc1 -fmodules \
// RUN:   -fmodule-map-file=%S/Inputs/pch-config-macros/include/module.modulemap \
// RUN:   -fmodules-cache-path=%t/cache -I %S/Inputs/pch-config-macros/include \
// RUN:   %t/h1.h -emit-pch -o %t/h1.h.pch -DCONFIG1 -DCONFIG2
// RUN: %clang_cc1 -fmodules \
// RUN:   -fmodule-map-file=%S/Inputs/pch-config-macros/include/module.modulemap \
// RUN:   -fmodules-cache-path=%t/cache -I %S/Inputs/pch-config-macros/include \
// RUN:   -include-pch %t/h1.h.pch %t/bridging.h -emit-pch -o %t/bridging.h.pch \
// RUN:   -DCONFIG1 -DCONFIG2
// RUN: %clang_cc1 -fmodules \
// RUN:   -fmodule-map-file=%S/Inputs/pch-config-macros/include/module.modulemap \
// RUN:   -fmodules-cache-path=%t/cache -I %S/Inputs/pch-config-macros/include \
// RUN:   -emit-obj -o %t/main.o %t/main.c -include-pch %t/bridging.h.pch \
// RUN:   -DCONFIG1 -DCONFIG2 -verify

// Checking that the warnings fire correctly when we compile with a chain of
// PCHs.
// RUN: %clang_cc1 -fmodules \
// RUN:   -fmodule-map-file=%S/Inputs/pch-config-macros/include/module.modulemap \
// RUN:   -fmodules-cache-path=%t/cache -I %S/Inputs/pch-config-macros/include \
// RUN:   -emit-obj -o %t/compile_warning_1.o %t/compile_warning_1.c -include-pch \
// RUN:    %t/bridging.h.pch -DCONFIG1 -DCONFIG2 -verify
// RUN: %clang_cc1 -fmodules \
// RUN:   -fmodule-map-file=%S/Inputs/pch-config-macros/include/module.modulemap \
// RUN:   -fmodules-cache-path=%t/cache -I %S/Inputs/pch-config-macros/include \
// RUN:   -emit-obj -o %t/compile_warning_2.o %t/compile_warning_2.c -include-pch \
// RUN:    %t/bridging.h.pch -DCONFIG1 -DCONFIG2 -verify
// RUN: %clang_cc1 -fmodules \
// RUN:   -fmodule-map-file=%S/Inputs/pch-config-macros/include/module.modulemap \
// RUN:   -fmodules-cache-path=%t/cache -I %S/Inputs/pch-config-macros/include \
// RUN:   -emit-obj -o %t/compile_warning_3.o %t/compile_warning_3.c -include-pch \
// RUN:    %t/bridging.h.pch -DCONFIG1 -DCONFIG2 -verify

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

// Checks against expected warnings.
//--- compile_warning_1.c
#undef CONFIG1 // expected-note{{macro was #undef'd here}}
#include "Mod1.h" // expected-warning{{#undef of configuration macro 'CONFIG1' has no effect on the import of 'Mod1'; pass '-UCONFIG1' on the command line to configure the module}}

int main_func() {
    return foo() + bar();
}

//--- compile_warning_2.c
#define CONFIG3  // expected-note{{macro was defined here}}
#include "Mod1.h" // expected-warning{{definition of configuration macro 'CONFIG3' has no effect on the import of 'Mod1'; pass '-DCONFIG3=...' on the command line to configure the module}}

int main_func() {
    return foo() + bar();
}

//--- compile_warning_3.c
#define CONFIG1 2 // expected-warning{{'CONFIG1' macro redefined}} expected-note{{previous definition is here}} expected-note{{macro was defined here}}
#include "Mod1.h" // expected-warning{{definition of configuration macro 'CONFIG1' has no effect on the import of 'Mod1'; pass '-DCONFIG1=...' on the command line to configure the module}}

int main_func() {
    return foo() + bar();
}
