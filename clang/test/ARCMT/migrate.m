// RUN: rm -rf %t
// RUN: %clang_cc1 -arcmt-migrate -mt-migrate-directory %t %S/Inputs/test1.m.in -x objective-c 
// RUN: %clang_cc1 -arcmt-migrate -mt-migrate-directory %t %S/Inputs/test2.m.in -x objective-c 
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %S/Inputs/test1.m.in.result %S/Inputs/test2.m.in.result %S/Inputs/test.h.result
// RUN: rm -rf %t
// DISABLE: mingw32
