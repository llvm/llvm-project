void g1(int * _Nonnull);

void g2(int (^block)(int, int)); // expected-warning{{block pointer is missing a nullability type specifier}}

void g3(const
        id // expected-warning{{missing a nullability type specifier}}
        volatile
        * // expected-warning{{missing a nullability type specifier}}
        ); 

@interface SomeClass
@property (retain,nonnull) id property1;
@property (retain,nullable) SomeClass *property2;
- (nullable SomeClass *)method1;
- (void)method2:(nonnull SomeClass *)param;
@property (readonly, weak) SomeClass *property3; // expected-warning{{missing a nullability type specifier}}
@end

@interface SomeClass ()
@property (readonly, weak) SomeClass *property4; // expected-warning{{missing a nullability type specifier}}
@end
