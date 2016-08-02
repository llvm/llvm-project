// RUN: rm -rf %t
// RUN: %clang_cc1 -objcmt-migrate-readwrite-property -objcmt-migrate-instancetype %s -triple x86_64-apple-darwin11 -migrate -o %t.remap
// RUN: c-arcmt-test %t.remap | arcmt-test -verify-transformed-files %S/header1.h.result %S/header2.h.result
// RUN: %clang_cc1 -objcmt-migrate-readwrite-property -objcmt-migrate-instancetype -objcmt-white-list-dir-path=%S/Inputs %s -triple x86_64-apple-darwin11 -migrate -o %t.remap
// RUN: c-arcmt-test %t.remap | arcmt-test -verify-transformed-files %S/header1.h.result

@interface NSObject
+ (id)alloc;
@end

#include "header1.h"
#include "header2.h"
