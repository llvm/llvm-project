// Make sure we don't crash when serializing a PCH with an include from a
// modulemap file in nomodules mode.
// No need to pass -fno-modules explicitly, absence implies negation for cc1.
// RUN: %clang_cc1 -I %S/Inputs/modulemaps-nomodules -fmodule-map-file=%S/Inputs/modulemaps-nomodules/module.modulemap %s -emit-pch -o /dev/null

#include "header.h"
