// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-runtime-has-weak -fsyntax-only -fobjc-arc -verify %s
// rdar://9340606

@interface Foo {
@public
    id __unsafe_unretained x;
    id __weak y;
    id __autoreleasing z; // expected-error {{instance variables cannot have __autoreleasing ownership}}
}
@property(strong) id x;
@property(strong) id y;
@property(strong) id z;
@end

@interface Bar {
@public
    id __unsafe_unretained x;
    id __weak y;
    id __autoreleasing z; // expected-error {{instance variables cannot have __autoreleasing ownership}}
}
@property(retain) id x;
@property(retain) id y;
@property(retain) id z;
@end

@interface Bas {
@public
    id __unsafe_unretained x;
    id __weak y;
    id __autoreleasing z; // expected-error {{instance variables cannot have __autoreleasing ownership}}
}
@property(copy) id x;
@property(copy) id y;
@property(copy) id z;
@end

// Errors should start about here :-)

@interface Bat 
@property(strong) __unsafe_unretained id x; // expected-error {{strong property 'x' may not also be declared __unsafe_unretained}}
@property(strong) __weak id y; // expected-error {{strong property 'y' may not also be declared __weak}}
@property(strong) __autoreleasing id z; // expected-error {{strong property 'z' may not also be declared __autoreleasing}}
@end

@interface Bau
@property(retain) __unsafe_unretained id x; // expected-error {{strong property 'x' may not also be declared __unsafe_unretained}}
@property(retain) __weak id y; // expected-error {{strong property 'y' may not also be declared __weak}}
@property(retain) __autoreleasing id z; // expected-error {{strong property 'z' may not also be declared __autoreleasing}}
@end

@interface Bav 
@property(copy) __unsafe_unretained id x; // expected-error {{strong property 'x' may not also be declared __unsafe_unretained}}
@property(copy) __weak id y; // expected-error {{strong property 'y' may not also be declared __weak}}
@property(copy) __autoreleasing id z; // expected-error {{strong property 'z' may not also be declared __autoreleasing}}
@end

@interface Bingo 
@property(assign) __unsafe_unretained id x;
@property(assign) __weak id y; // expected-error {{unsafe_unretained property 'y' may not also be declared __weak}}
@property(assign) __autoreleasing id z; // expected-error {{unsafe_unretained property 'z' may not also be declared __autoreleasing}}
@end

@interface Batman 
@property(unsafe_unretained) __unsafe_unretained id x;
@property(unsafe_unretained) __weak id y; // expected-error {{unsafe_unretained property 'y' may not also be declared __weak}}
@property(unsafe_unretained) __autoreleasing id z; // expected-error {{unsafe_unretained property 'z' may not also be declared __autoreleasing}}
@end

// rdar://9396329
@interface Super
@property (readonly, retain) id foo;
@property (readonly, weak) id fee;
@property (readonly, strong) id frr;
@end

@interface Bugg : Super
@property (readwrite) id foo;
@property (readwrite) id fee;
@property (readwrite) id frr;
@end

// rdar://20152386
// rdar://20383235

@interface NSObject @end

#pragma clang assume_nonnull begin
@interface I: NSObject
@property(nonatomic, weak) id delegate; // Do not warn, nullable is inferred. 
@property(nonatomic, weak, readonly) id ROdelegate; // Do not warn, nullable is inferred.
@property(nonatomic, weak, nonnull) id NonNulldelete; // expected-error {{property attributes 'nonnull' and 'weak' are mutually exclusive}}
@property(nonatomic, weak, nullable) id Nullabledelete; // do not warn

// strong cases.
@property(nonatomic, strong) id stdelegate; // Do not warn
@property(nonatomic, readonly) id stROdelegate; // Do not warn
@property(nonatomic, strong, nonnull) id stNonNulldelete; // Do not warn
@property(nonatomic, nullable) id stNullabledelete; // do not warn
@end
#pragma clang assume_nonnull end

@interface J: NSObject
@property(nonatomic, weak) id ddd;   // Do not warn, nullable is inferred.
@property(nonatomic, weak, nonnull) id delegate; // expected-error {{property attributes 'nonnull' and 'weak' are mutually exclusive}}
@property(nonatomic, weak, nonnull, readonly) id ROdelegate; // expected-error {{property attributes 'nonnull' and 'weak' are mutually exclusive}}
@end

