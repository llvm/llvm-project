// RUN: %clang_cc1  -fsyntax-only -verify %s
// RUN: %clang_cc1  -fsyntax-only -triple i386-apple-macosx10.9.0 -fobjc-runtime=macosx-fragile-10.9.0 -fobjc-subscripting-legacy-runtime -verify %s

#define nil ((void *)0)

void checkNSDictionaryUnavailableDiagnostic(void) {
  id key;
  id value;
  id dict = @{ key : value }; // expected-error {{definition of class NSDictionary must be available to use Objective-C dictionary literals}}
}

@class NSDictionary; // expected-note {{forward declaration of class here}}

void checkNSDictionaryFDDiagnostic(void) {
  id key;
  id value;
  id dic = @{ key : value }; // expected-error {{definition of class NSDictionary must be available to use Objective-C dictionary literals}}
}

@interface NSNumber
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithInt:(int)value;
@end

@protocol NSCopying @end
typedef unsigned long NSUInteger;
typedef long NSInteger;

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id <NSCopying> [])keys count:(NSUInteger)cnt;
- (void)setObject:(id)object forKeyedSubscript:(id)key;
- (id)objectForKeyedSubscript:(id)key;
@end

@interface NSString<NSCopying>
@end

@interface NSArray
- (id)objectAtIndexedSubscript:(NSInteger)index;
- (void)setObject:(id)object atIndexedSubscript:(NSInteger)index;
@end

void *pvoid;
int main(void) {
	NSDictionary *dict = @{ @"name":@666 };
        dict[@"name"] = @666;

        dict["name"] = @666; // expected-error {{indexing expression is invalid because subscript type 'char *' is not an Objective-C pointer}}

        [@{@"foo" : @"bar"} objectForKeyedSubscript:nil];
        (void)@{@"foo" : @"bar"}[nil];
        [@{@"foo" : @"bar"} objectForKeyedSubscript:pvoid];
        (void)@{@"foo" : @"bar"}[pvoid];

	[@{@"foo" : @"bar"} setObject:nil forKeyedSubscript:@"gorf"];
        @{@"foo" : @"bar"}[nil] = @"gorf";
	[@{@"foo" : @"bar"} setObject:pvoid forKeyedSubscript:@"gorf"];
        @{@"foo" : @"bar"}[pvoid] = @"gorf";

	return 0;
}

enum XXXYYYZZZType { XXXYYYZZZTypeAny }; // expected-note {{'XXXYYYZZZTypeAny' declared here}}
void foo(void) {
  NSDictionary *d = @{
    @"A" : @(XXXYYYZZZTypeA), // expected-error {{use of undeclared identifier 'XXXYYYZZZTypeA'; did you mean 'XXXYYYZZZTypeAny'}}
    @"F" : @(XXXYYYZZZTypeSomethingSomething), // expected-error {{use of undeclared identifier 'XXXYYYZZZTypeSomethingSomething'}}
  };
}
