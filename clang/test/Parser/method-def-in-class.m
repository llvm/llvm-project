// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://7029784

@interface A
-(id) f0 { // expected-error {{expected ';' after method prototype}}
  assert(0);
}
@end

@interface C
- (id) f0 { // expected-error {{expected ';' after method prototype}}
    assert(0);
};
@end
