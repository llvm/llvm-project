// RUN: %clang_cc1 %s -verify -fno-builtin

_Static_assert(__has_feature(attribute_diagnose_if_objc), "feature check failed?");

#define _diagnose_if(...) __attribute__((diagnose_if(__VA_ARGS__)))

@interface I
-(void)meth _diagnose_if(1, "don't use this", "warning"); // expected-note 1{{from 'diagnose_if'}}
@property (assign) id prop _diagnose_if(1, "don't use this", "warning"); // expected-note 2{{from 'diagnose_if'}}
@end

void test(I *i) {
  [i meth]; // expected-warning {{don't use this}}
  id o1 = i.prop; // expected-warning {{don't use this}}
  id o2 = [i prop]; // expected-warning {{don't use this}}
}
