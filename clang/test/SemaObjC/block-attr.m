// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -fblocks -fobjc-gc-only %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fsyntax-only -verify -fblocks -fobjc-gc-only %s

@interface Thing  {}

@property void(^someBlock)(void); // expected-warning {{'copy' attribute must be specified for the block property}}
@property(copy)  void(^OK)(void);

@property (readonly) void (^block)(void); // readonly property is OK

@end
