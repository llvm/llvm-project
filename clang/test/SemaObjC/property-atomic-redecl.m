// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s

@interface A
@end

// Readonly, atomic public redeclaration of property in subclass.
@interface AtomicInheritanceSuper
@property (readonly) A *property;
@end

@interface AtomicInheritanceSuper()
@property (nonatomic,readwrite,retain) A *property;
@end

@interface AtomicInheritanceSub : AtomicInheritanceSuper
@property (readonly) A *property;
@end

// Readonly, atomic public redeclaration of property in subclass.
@interface AtomicInheritanceSuper2
@property (readonly) A *property;
@end

@interface AtomicInheritanceSub2 : AtomicInheritanceSuper2
@property (nonatomic, readwrite, retain) A *property; // FIXME: should be okay
@end

@interface ReadonlyAtomic
@property (readonly, nonatomic) A *property; // expected-note{{property declared here}}
@end

@interface ReadonlyAtomic ()
@property (readwrite) A *property; // expected-warning{{'atomic' attribute on property 'property' does not match the property inherited from 'ReadonlyAtomic'}}
@end

// Readonly, atomic public redeclaration of property in subclass.
@interface AtomicInheritanceSuper3
@property (readonly,atomic) A *property; // expected-note{{property declared here}}
@end

@interface AtomicInheritanceSuper3()
@property (nonatomic,readwrite,retain) A *property; // expected-warning{{'atomic' attribute on property 'property' does not match the property inherited from 'AtomicInheritanceSuper3'}}
@end

@interface AtomicInheritanceSub3 : AtomicInheritanceSuper3
@property (readonly) A *property;
@end

// Readonly, atomic public redeclaration of property in subclass.
@interface AtomicInheritanceSuper4
@property (readonly, atomic) A *property; // expected-note{{property declared here}}
@end

@interface AtomicInheritanceSub4 : AtomicInheritanceSuper4
@property (nonatomic, readwrite, retain) A *property; // expected-warning{{atomic' attribute on property 'property' does not match the property inherited from 'AtomicInheritanceSuper4'}}
@end

