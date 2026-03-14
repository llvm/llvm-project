// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
extern "C" {
@class Protocol;
}

extern "C" {
@class I;
}

@interface I
@end

@protocol VKAnnotation;
extern "C" {

@protocol VKAnnotation
  @property (nonatomic, assign) id coordinate;
@end
}
