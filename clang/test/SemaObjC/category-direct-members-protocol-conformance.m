// RUN: %clang_cc1 -fsyntax-only -verify %s

__attribute__((objc_root_class))
@interface RootClass

- (void)baseMethod;

@end

__attribute__((objc_direct_members))
@interface I : RootClass

- (void)direct; // expected-note {{direct member declared here}}

@end

@protocol P
- (void)direct;
@end

@interface I (Cat1) <P> // expected-error {{category 'Cat1' cannot conform to protocol 'P' because of direct members declared in interface 'I'}}
@end

@protocol BaseP
- (void)baseMethod;
@end

@interface I (CatBase) <BaseP> // OK
@end

@protocol P2
- (void)indirect;
@end

@interface I (Cat2) <P2> // OK
- (void)indirect;
@end

@protocol P3
- (void)indirect3;
@end

@interface I (Cat3) <P3> // OK
@end

@interface ExpDirect : RootClass

- (void)direct __attribute__((objc_direct)); // expected-note {{direct member declared here}}

- (void)directRecursive __attribute__((objc_direct)); // expected-note {{direct member declared here}}

@end

@interface ExpDirect (CatExpDirect) <P> // expected-error {{category 'CatExpDirect' cannot conform to protocol 'P' because of direct members declared in interface 'ExpDirect'}}
@end

@protocol PRecursive1
- (void)directRecursive;
@end

@protocol PRecursiveTop <PRecursive1>
@end

@interface ExpDirect () <PRecursiveTop> // expected-error {{class extension cannot conform to protocol 'PRecursive1' because of direct members declared in interface 'ExpDirect'}}
@end


@protocol PProp

@property (nonatomic, readonly) I *name;

@end

__attribute__((objc_direct_members))
@interface IProp1 : RootClass

@property (nonatomic, readonly) I *name; // expected-note {{direct member declared here}}

@end

@interface IProp1 () <PProp> // expected-error {{class extension cannot conform to protocol 'PProp' because of direct members declared in interface 'IProp1'}}
@end


@protocol PProp2

@property (nonatomic, readonly, class) I *name;

@end

@interface IProp2 : RootClass

@property (nonatomic, readonly, class, direct) I *name; // expected-note {{direct member declared here}}

@end

@interface IProp2 () <PProp2> // expected-error {{class extension cannot conform to protocol 'PProp2' because of direct members declared in interface 'IProp2'}}
@end
