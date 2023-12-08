// RUN: %clang_cc1 -objcmt-migrate-designated-init -objcmt-migrate-readwrite-property -objcmt-migrate-instancetype -x objective-c %S/file1.m.in -triple x86_64-apple-darwin11 -fobjc-arc -migrate -o %t1.remap
// RUN: %clang_cc1 -objcmt-migrate-designated-init -objcmt-migrate-readwrite-property -objcmt-migrate-instancetype -x objective-c %S/file2.m.in -triple x86_64-apple-darwin11 -fobjc-arc -migrate -o %t2.remap
// RUN: c-arcmt-test %t1.remap %t2.remap | arcmt-test -verify-transformed-files %S/header1.h.result %S/file2.m.in.result
