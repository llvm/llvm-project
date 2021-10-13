// rm -rf %t && mkdir %t

// RUN: %clang_cc1 -fmodules -emit-module -fmodule-name=mod %S/Inputs/pch-shared-module/module.modulemap -o %t/mod.pcm

// RUN: %clang_cc1 -fmodules -emit-pch %S/Inputs/pch-shared-module/pch.h -o %t/pch.h.gch \
// RUN:   -fmodule-file=%t/mod.pcm -fmodule-map-file=%S/Inputs/pch-shared-module/module.modulemap

// Check that `mod.pcm` is loaded correctly, even though it's imported by the PCH as well.
// RUN: %clang_cc1 -fmodules -fsyntax-only %s -include-pch %t/pch.h.gch -I %S/Inputs/pch-shared-module \
// RUN:   -fmodule-file=%t/mod.pcm -fmodule-map-file=%S/Inputs/pch-shared-module/module.modulemap -verify

#include "mod.h"

// expected-no-diagnostics
