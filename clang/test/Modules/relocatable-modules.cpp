// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header hu-01.h \
// RUN:   -fmodule-name=hu-01 -o hu-01.pcm

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header hu-02.h \
// RUN:  -Wno-experimental-header-units -fmodule-file=hu-01.pcm -o hu-02-abs.pcm

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header hu-02.h \
// RUN:  -Wno-experimental-header-units -fmodule-file=hu-01.pcm -o hu-02-rel.pcm \
// RUN:  -fmodule-file-home-is-cwd

// RUN: %clang -module-file-info hu-02-abs.pcm | FileCheck %s --check-prefix=IMPORT-ABS -DPREFIX=%t
// IMPORT-ABS: Imports module 'hu-01': [[PREFIX]]{{/|\\}}hu-01.pcm

// RUN: %clang -module-file-info hu-02-rel.pcm | FileCheck %s --check-prefix=IMPORT-REL
// IMPORT-REL: Imports module 'hu-01': hu-01.pcm

// RUN: llvm-bcanalyzer --dump --disable-histogram %t/hu-02-abs.pcm \
// RUN:   | FileCheck %s --check-prefix=INPUT-ABS -DPREFIX=%t
// INPUT-ABS: <INPUT_FILE {{.*}}/> blob data = '[[PREFIX]]{{/|\\}}hu-02.h'

// RUN: llvm-bcanalyzer --dump --disable-histogram %t/hu-02-rel.pcm \
// RUN:   | FileCheck %s --check-prefix=INPUT-REL
// INPUT-REL: <INPUT_FILE {{.*}}/> blob data = 'hu-02.h'

//--- hu-01.h
inline void f() {}

//--- hu-02.h
import "hu-01.h";

inline void g() {
  f();
}

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.cppm -o %t/a-abs.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.cppm -o %t/a-rel.pcm \
// RUN:   -fmodule-file-home-is-cwd

// RUN: llvm-bcanalyzer --dump --disable-histogram %t/a-abs.pcm \
// RUN:   | FileCheck %s --check-prefix=M-INPUT-ABS -DPREFIX=%t
// M-INPUT-ABS: <INPUT_FILE {{.*}}/> blob data = '[[PREFIX]]{{/|\\}}a.cppm'

// RUN: llvm-bcanalyzer --dump --disable-histogram %t/a-rel.pcm \
// RUN:   | FileCheck %s --check-prefix=M-INPUT-REL
// M-INPUT-REL: <INPUT_FILE {{.*}}/> blob data = 'a.cppm'

//--- a.cppm
export module a;
