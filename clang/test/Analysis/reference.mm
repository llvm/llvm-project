// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify -Wno-null-dereference %s

@interface Foo
- (int &)ref;
@end

Foo *getFoo() { return 0; }

void testNullPointerSuppression() {
	getFoo().ref = 1;
}

void testPositiveNullReference() {
  Foo *x = 0;
	x.ref = 1; // expected-warning {{The receiver of message 'ref' is nil, which results in forming a null reference}}
}

