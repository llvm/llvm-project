// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface _Child // expected-note{{'_Child' declared here}}
+ (int) flashCache;
@end

@interface Child (Categ) // expected-error {{cannot find interface declaration for 'Child'; did you mean '_Child'?}}
+ (int) flushCache2;
@end

@implementation OtherChild (Categ) // expected-error {{cannot find interface declaration for 'OtherChild'}}
+ (int) flushCache2 { [super flashCache]; } // expected-error {{no @interface declaration found in class messaging of 'flushCache2'}}
@end
