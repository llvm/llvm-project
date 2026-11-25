// RUN: rm -rf %t
// RUN: mkdir -p %t/one %t/two
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -fmodule-name=hu-01 -emit-header-unit \
// RUN:   -xc++-user-header %t/hu-01.h -o %t/one/hu-01.pcm
// RUN: cp %t/one/hu-01.pcm %t/two/hu-01.pcm

// A header unit loaded through a named mapping does not serialize its PCM path.
// RUN: %clang_cc1 -std=c++20 -fmodule-name=hu-02 -emit-header-unit \
// RUN:   -xc++-user-header %t/hu-02.h -Wno-experimental-header-units \
// RUN:   -fmodule-map-file=%t/hu-01.map \
// RUN:   -fmodule-file=hu-01=%t/one/hu-01.pcm -o %t/hu-02-mapped.pcm
// RUN: cp %t/hu-02-mapped.pcm %t/hu-02-mapped-from-one.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-name=hu-02 -emit-header-unit \
// RUN:   -xc++-user-header %t/hu-02.h -Wno-experimental-header-units \
// RUN:   -fmodule-map-file=%t/hu-01.map \
// RUN:   -fmodule-file=hu-01=%t/two/hu-01.pcm -o %t/hu-02-mapped.pcm
// RUN: diff %t/hu-02-mapped-from-one.pcm %t/hu-02-mapped.pcm

// The dependency is loaded through its named mapping.
// RUN: %clang_cc1 -std=c++20 -module-file-info %t/hu-02-mapped.pcm \
// RUN:   -fmodule-file=hu-01=%t/two/hu-01.pcm \
// RUN:   | FileCheck %s --check-prefix=HU-MAPPED
// HU-MAPPED: Imports module 'hu-01':{{ *}}{{$}}

// A pathless dependency requires a matching -fmodule-file=<name>=<pcm>.
// RUN: not %clang_cc1 -std=c++20 -module-file-info %t/hu-02-mapped.pcm 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MISSING
// MISSING: error: failed to find module file for module 'hu-01'

// A named mapping must reference a PCM with the expected module name.
// RUN: %clang_cc1 -std=c++20 -fmodule-name=hu-other -emit-header-unit \
// RUN:   -xc++-user-header %t/hu-01.h -o %t/hu-other.pcm
// RUN: not %clang_cc1 -std=c++20 -module-file-info %t/hu-02-mapped.pcm \
// RUN:   -fmodule-file=hu-01=%t/hu-other.pcm 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MISMATCH
// MISMATCH: error: failed to find module file for module 'hu-01'

// An eagerly loaded header unit retains its PCM path even if a named mapping
// also exists.
// RUN: %clang_cc1 -std=c++20 -fmodule-name=hu-02 -emit-header-unit \
// RUN:   -xc++-user-header %t/hu-02.h -Wno-experimental-header-units \
// RUN:   -fmodule-map-file=%t/hu-01.map -fmodule-file=%t/one/hu-01.pcm \
// RUN:   -fmodule-file=hu-01=%t/one/hu-01.pcm -o %t/hu-02-eager.pcm
// RUN: %clang_cc1 -std=c++20 -module-file-info %t/hu-02-eager.pcm \
// RUN:   -fmodule-file=hu-01=%t/one/hu-01.pcm \
// RUN:   | FileCheck %s --check-prefix=HU-EAGER -DPREFIX=%t
// HU-EAGER: Imports module 'hu-01': [[PREFIX]]{{/|\\}}one{{/|\\}}hu-01.pcm

// The PCM paths of C++20 named module dependencies are not serialized.
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.cppm \
// RUN:   -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/foo.cppm \
// RUN:   -fmodule-file=A=%t/A.pcm \
// RUN:   -o %t/Foo.pcm
// RUN: %clang_cc1 -std=c++20 -module-file-info %t/Foo.pcm \
// RUN:   -fmodule-file=A=%t/A.pcm \
// RUN:   | FileCheck %s --check-prefix=NAMED-MODULE
// NAMED-MODULE: Module name: Foo
// NAMED-MODULE-NEXT: Imports module 'A':{{ *}}{{$}}

//--- hu-01.h
inline void f() {}

//--- hu-02.h
import "hu-01.h";

//--- hu-01.map
module "hu-01" {
  header "hu-01.h"
  export *
}

//--- a.cppm
export module A;

//--- foo.cppm
export module Foo;
import A;
