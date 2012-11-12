// RUN: rm -rf %t.migrate
// RUN: %clang_cc1 -arcmt-migrate -mt-migrate-directory %t.migrate %S/"with space"/test1.m.in -x objective-c 
// RUN: %clang_cc1 -arcmt-migrate -mt-migrate-directory %t.migrate %S/"with space"/test2.m.in -x objective-c 
// RUN: c-arcmt-test -mt-migrate-directory %t.migrate | arcmt-test -verify-transformed-files %S/"with space"/test1.m.in.result %S/"with space"/test2.m.in.result %S/"with space"/test.h.result
// RUN: rm -rf %t.migrate
// DISABLE: mingw32
