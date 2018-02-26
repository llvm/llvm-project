void g1(int * _Nonnull);

void g2(int (^block)(int, int)); // expected-warning{{block pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable' if the block pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the block pointer should never be null}}

void g3(const
        id // expected-warning{{missing a nullability type specifier}}
        volatile
// expected-note@-1 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the pointer should never be null}}
        * // expected-warning{{missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the pointer should never be null}}
        ); 

@interface SomeClass
@property (retain,nonnull) id property1;
@property (retain,nullable) SomeClass *property2;
- (nullable SomeClass *)method1;
- (void)method2:(nonnull SomeClass *)param;
@property (readonly, weak) SomeClass *property3; // expected-warning{{missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the pointer should never be null}}
@end

@interface SomeClass ()
@property (readonly, weak) SomeClass *property4; // expected-warning{{missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the pointer should never be null}}
@end
