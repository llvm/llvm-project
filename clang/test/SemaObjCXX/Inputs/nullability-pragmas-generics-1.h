#pragma clang assume_nonnull begin

__attribute__((objc_root_class))
@interface B
@end

@interface C : B
@end

__attribute__((objc_root_class))
@interface NSGeneric<T : B *> // expected-note{{type parameter 'T' declared here}}
- (T)tee;
- (nullable T)maybeTee;
@end

typedef NSGeneric<C *> *Generic_with_C;

#pragma clang assume_nonnull end

@interface NSGeneric<T : C *>(Blah) // expected-error{{type bound 'C *' for type parameter 'T' conflicts with previous bound 'B *'}}
@end
