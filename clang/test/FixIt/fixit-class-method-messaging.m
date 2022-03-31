// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// rdar://16263395

@interface NSObject @end

@interface I : NSObject // expected-note 3 {{receiver is instance of class declared here}}
+ (id) ClassMeth;
- (I*) MethInstPI;
@end

I* pi;

I* foobar(void);

@implementation I
- (id) PrivInstMeth {
  [ foobar() ClassMeth]; // expected-warning {{instance method '-ClassMeth' not found (return type defaults to 'id')}} \
			 // expected-note {{receiver expression is here}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:5-[[@LINE-2]]:13}:"I
  [[self MethInstPI] ClassMeth]; // expected-warning {{instance method '-ClassMeth' not found (return type defaults to 'id')}} \
				 // expected-note {{receiver expression is here}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:4-[[@LINE-2]]:21}:"I
  return [pi ClassMeth]; // expected-warning {{instance method '-ClassMeth' not found (return type defaults to 'id')}} \
                         // expected-note {{receiver expression is here}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:11-[[@LINE-2]]:13}:"I
}
+ (id) ClassMeth { return 0; }
- (I*) MethInstPI { return 0; }
@end
