// Test objc_boxable update record

// RUN: %clang_cc1 -x objective-c %S/objc_boxable_record.h -emit-pch -o %t1
// RUN: %clang_cc1 -x objective-c %S/objc_boxable_record_attr.h -include-pch %t1 -emit-pch -o %t2
// RUN: %clang_cc1 %s -include-pch %t2 -fsyntax-only -verify

// expected-no-diagnostics 

__attribute__((objc_root_class))
@interface NSValue
+ (NSValue *)valueWithBytes:(const void *)bytes objCType:(const char *)type;
@end

void doStuff(struct boxable b) {
  id v = @(b);
}

