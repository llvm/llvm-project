// RUN: %clang_cc1 -fsyntax-only -Wreceiver-forward-class -verify %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only  -Wreceiver-forward-class -verify %s

@class A; // expected-note {{forward declaration of class here}}

@interface B
-(int) width; // expected-note {{using}}
@end
@interface C
-(float) width; // expected-note {{also found}}
@end

int f0(A *x) {
  return [x width]; // expected-warning {{receiver type 'A' for instance message is a forward declaration}} \
                    // expected-warning {{multiple methods named 'width' found}} \
                    // expected-note {{receiver is treated with 'id' type for purpose of method lookup}}
}

