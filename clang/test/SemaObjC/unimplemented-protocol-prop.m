// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class -disable-objc-default-synthesize-properties %s

@protocol PROTOCOL0
@required
@property float MyProperty0; // expected-note 2 {{property declared}}
@end

@protocol PROTOCOL<PROTOCOL0>
@required
@property float MyProperty; // expected-note 2 {{property declared}}
@optional
@property float OptMyProperty;
@end

@interface I <PROTOCOL>
@end

@implementation I @end // expected-warning {{property 'MyProperty0' requires method 'MyProperty0' to be defined}} \
                       // expected-warning {{property 'MyProperty0' requires method 'setMyProperty0:' to be defined}}\
                       // expected-warning {{property 'MyProperty' requires method 'MyProperty' to be defined}} \
                       // expected-warning {{property 'MyProperty' requires method 'setMyProperty:' to be defined}}

// property is implemented in super class. No warning

@protocol PROTOCOL1
@property int MyProp;
@end

@interface superclass
@property int MyProp;
@end

@interface childclass : superclass <PROTOCOL1>
@end

@implementation childclass
@end

