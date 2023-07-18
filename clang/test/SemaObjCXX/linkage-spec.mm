// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
extern "C" {
@class Protocol;
}

// <rdar://problem/7827709>
extern "C" {
@class I;
}

@interface I
@end

// rdar://10015110
@protocol VKAnnotation;
extern "C" {

@protocol VKAnnotation
  @property (nonatomic, assign) id coordinate;
@end
}
