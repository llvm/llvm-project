// RUN: rm -rf %t.migrate
// RUN: %clang_cc1 -arcmt-action=migrate -mt-migrate-directory %t.migrate %S/Inputs/"with space"/test1.m.in -x objective-c
// RUN: %clang_cc1 -arcmt-action=migrate -mt-migrate-directory %t.migrate %S/Inputs/"with space"/test2.m.in -x objective-c
// RUN: c-arcmt-test -mt-migrate-directory %t.migrate | arcmt-test -verify-transformed-files %S/Inputs/"with space"/test1.m.in.result %S/Inputs/"with space"/test2.m.in.result %S/Inputs/"with space"/test.h.result
// RUN: rm -rf %t.migrate
