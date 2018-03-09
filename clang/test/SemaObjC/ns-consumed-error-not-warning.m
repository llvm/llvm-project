// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -fobjc-arc -verify -fblocks -triple x86_64-apple-darwin10.0.0 -DOBJCARC %s
// rdar://36265651

@interface A
-(void) m:(id)p; // expected-note {{parameter declared here}}
@end

@interface B : A
-(void) m:(__attribute__((ns_consumed)) id)p; // expected-error {{overriding method has mismatched ns_consumed attribute on its parameter}}
@end

@import empty;
