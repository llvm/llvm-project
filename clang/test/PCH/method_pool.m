// Test this without pch.
// RUN: %clang_cc1 -include %S/method_pool.h -fsyntax-only -verify -Wno-objc-root-class %s

// Test with pch.
// RUN: %clang_cc1 -x objective-c -Wno-objc-root-class -emit-pch -o %t %S/method_pool.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify -Wno-objc-root-class %s 

int message_id(id x) {
   return [x instMethod:17]; // expected-warning{{multiple methods}}
}

/* expected-note@method_pool.h:17{{using}} */
/* expected-note@method_pool.h:21{{also}} */
