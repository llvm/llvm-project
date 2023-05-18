// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c %S/Common.h -emit-pch -o %t.pch
// RUN: %clang_cc1 -arcmt-action=migrate -mt-migrate-directory %t %S/Inputs/test1.m.in -x objective-c -include-pch %t.pch
// RUN: %clang_cc1 -arcmt-action=migrate -mt-migrate-directory %t %S/Inputs/test2.m.in -x objective-c -include-pch %t.pch
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %S/Inputs/test1.m.in.result %S/Inputs/test2.m.in.result %S/Inputs/test.h.result
// RUN: rm -rf %t
