// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// A PCM whose paths are relative to the consumer's working directory should
// load successfully even though it does not contain a MODULE_DIRECTORY record.
// RUN: %clang_cc1 -emit-module -x objective-c -fmodules \
// RUN:   -fno-implicit-modules -fmodule-file-home-is-cwd \
// RUN:   -fmodule-name=Repro mod/module.modulemap -o Repro-relative.pcm
// RUN: %clang_cc1 -fsyntax-only -x objective-c -fmodules \
// RUN:   -fno-implicit-modules -fmodule-file-home-is-cwd \
// RUN:   -fmodule-file=Repro=Repro-relative.pcm use.m

// An explicitly loaded PCM should also not crash if its absolute module home
// directory has been removed.
// RUN: %clang_cc1 -emit-module -x objective-c -fmodules \
// RUN:   -fno-implicit-modules -fmodule-name=Repro \
// RUN:   mod/module.modulemap -o Repro-absolute.pcm
// RUN: rm -rf mod
// RUN: %clang_cc1 -fsyntax-only -x objective-c -fmodules \
// RUN:   -fno-implicit-modules -fmodule-file=Repro=Repro-absolute.pcm use.m

//--- mod/module.modulemap
module Repro {
  umbrella header "Repro.h"
  export *
  module * { export * }
}

//--- mod/Repro.h
#include "sub.h"

//--- mod/sub.h
static inline int repro_answer(void) { return 42; }

//--- use.m
@import Repro;
int main(void) { return repro_answer(); }
