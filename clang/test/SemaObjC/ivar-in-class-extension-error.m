// RUN: %clang_cc1 -fobjc-runtime=macosx-fragile-10.5 -fsyntax-only -verify %s

@interface A @end

@interface A () { 
  int _p0; // expected-error {{instance variables may not be placed in class extension}}
}
@property int p0;
@end

@interface A(CAT) { 
  int _p1; // expected-error {{instance variables may not be placed in categories}}
}
@end
