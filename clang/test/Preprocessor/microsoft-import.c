// RUN: %clang_cc1 -E -fms-compatibility %s 2>&1 | grep 'doh.c:100:2: error: #import of type library is an unsupported Microsoft feature'
// RUN: %clang_cc1 -E -fms-compatibility %s 2>&1 | grep 'doh.c:200:2: error: #import of type library is an unsupported Microsoft feature'
// RUN: %clang_cc1 -E -fms-compatibility %s 2>&1 | grep 'doh.c:300:2: error: #import of type library is an unsupported Microsoft feature'

#line 100 "doh.c"
#import "pp-record.h" // expected-error {{#import of type library is an unsupported Microsoft feature}}

// Test attributes
#line 200 "doh.c"
#import "pp-record.h" no_namespace, auto_rename // expected-error {{#import of type library is an unsupported Microsoft feature}}

// This will also fire the "#import of type library is an unsupported Microsoft feature"
// error, but we can't use -verify because there's no way to put the comment on the proper line
#line 300 "doh.c"
#import "pp-record.h" no_namespace \
                      auto_rename \
                      auto_search
