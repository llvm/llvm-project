// This test checks that a module map with a textual header can be marked as
// non-affecting if its header did not get included.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

//--- A.modulemap
module A { textual header "A.h" }
//--- B.modulemap
module B { header "B.h" }
//--- A.h
//--- B.h

// RUN: %clang_cc1 -fmodules -emit-module %t/B.modulemap -fmodule-name=B -o %t/B0.pcm \
// RUN:   -fmodule-map-file=%t/A.modulemap -fmodule-map-file=%t/B.modulemap

// RUN: %clang_cc1 -fmodules -emit-module %t/B.modulemap -fmodule-name=B -o %t/B1.pcm \
// RUN:                                    -fmodule-map-file=%t/B.modulemap

// RUN: diff %t/B0.pcm %t/B1.pcm
