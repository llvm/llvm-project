// RUN: %clang_cc1 -fsyntax-only -verify -Wselector-type-mismatch %s

__attribute__((objc_root_class))
@interface Inteface_Implementation
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly, direct) int direct_normal;
@property(nonatomic, readonly) int normal_direct; // expected-note {{previous declaration is here}}
@property(nonatomic, readonly, direct) int direct_direct;
@end

@implementation Inteface_Implementation
- (int)normal_normal {
  return 42;
}
- (int)direct_normal {
  return 42;
}
- (int)normal_direct __attribute__((objc_direct)) { // expected-error {{direct method implementation was previously declared not direct}}
  return 42;
}
- (int)direct_direct __attribute__((objc_direct)) {
  return 42;
}
@end

__attribute__((objc_root_class))
@interface Inteface_Extension
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly, direct) int direct_normal;
@property(nonatomic, readonly) int normal_direct;
@property(nonatomic, readonly, direct) int direct_direct;
@end

@interface Inteface_Extension ()
@property(nonatomic, readwrite) int normal_normal;
@property(nonatomic, readwrite) int direct_normal;
@property(nonatomic, readwrite, direct) int normal_direct;
@property(nonatomic, readwrite, direct) int direct_direct;
@end

@implementation Inteface_Extension
@end

__attribute__((objc_root_class))
@interface Extension_Implementation
@end

@interface Extension_Implementation ()
@property(nonatomic, readwrite) int normal_normal;
@property(nonatomic, readwrite, direct) int direct_normal;
@property(nonatomic, readwrite) int normal_direct; // expected-note {{previous declaration is here}}
@property(nonatomic, readwrite, direct) int direct_direct;
@end

@implementation Extension_Implementation
- (int)normal_normal {
  return 42;
}
- (int)direct_normal {
  return 42;
}
- (int)normal_direct __attribute__((objc_direct)) { // expected-error {{direct method implementation was previously declared not direct}}
  return 42;
}
- (int)direct_direct __attribute__((objc_direct)) {
  return 42;
}
@end

__attribute__((objc_root_class))
@interface Inteface_Category
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly, direct) int direct_normal; // expected-note {{previous declaration is here}}
@property(nonatomic, readonly) int normal_direct;         // expected-note {{previous declaration is here}}
@property(nonatomic, readonly, direct) int direct_direct; // expected-note {{previous declaration is here}}
@end

@interface Inteface_Category (SomeCategory)
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly) int direct_normal;         // expected-error {{property declaration conflicts with previous direct declaration of property 'direct_normal'}}
@property(nonatomic, readonly, direct) int normal_direct; // expected-error {{direct property declaration conflicts with previous declaration of property 'normal_direct'}}
@property(nonatomic, readonly, direct) int direct_direct; // expected-error {{direct property declaration conflicts with previous direct declaration of property 'direct_direct'}}
@end

@implementation Inteface_Category
@end

__attribute__((objc_root_class))
@interface Extension_Category
@end

@interface Extension_Category ()
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly, direct) int direct_normal; // expected-note {{previous declaration is here}}
@property(nonatomic, readonly) int normal_direct;         // expected-note {{previous declaration is here}}
@property(nonatomic, readonly, direct) int direct_direct; // expected-note {{previous declaration is here}}
@end

@interface Extension_Category (SomeCategory)
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly) int direct_normal;         // expected-error {{property declaration conflicts with previous direct declaration of property 'direct_normal'}}
@property(nonatomic, readonly, direct) int normal_direct; // expected-error {{direct property declaration conflicts with previous declaration of property 'normal_direct'}}
@property(nonatomic, readonly, direct) int direct_direct; // expected-error {{direct property declaration conflicts with previous direct declaration of property 'direct_direct'}}
@end

@implementation Extension_Category
@end

__attribute__((objc_root_class))
@interface Implementation_Category
@end

@interface Implementation_Category (SomeCategory)
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly, direct) int direct_normal; // expected-note {{previous declaration is here}}
@property(nonatomic, readonly) int normal_direct;         // expected-note {{previous declaration is here}}
@property(nonatomic, readonly, direct) int direct_direct; // expected-note {{previous declaration is here}}
@end

@implementation Implementation_Category
- (int)normal_normal {
  return 42;
}
- (int)direct_normal { // expected-error {{direct method was declared in a category but is implemented in the primary interface}}
  return 42;
}
- (int)normal_direct __attribute__((objc_direct)) { // expected-error {{direct method was declared in a category but is implemented in the primary interface}}
  return 42;
}
- (int)direct_direct __attribute__((objc_direct)) { // expected-error {{direct method was declared in a category but is implemented in the primary interface}}
  return 42;
}
@end

__attribute__((objc_root_class))
@interface Category_Category
@end

@interface Category_Category (SomeCategory)
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly, direct) int direct_normal; // expected-note {{previous declaration is here}}
@property(nonatomic, readonly) int normal_direct;         // expected-note {{previous declaration is here}}
@property(nonatomic, readonly, direct) int direct_direct; // expected-note {{previous declaration is here}}
@end

@interface Category_Category (SomeOtherCategory)
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly) int direct_normal;         // expected-error {{property declaration conflicts with previous direct declaration of property 'direct_normal'}}
@property(nonatomic, readonly, direct) int normal_direct; // expected-error {{direct property declaration conflicts with previous declaration of property 'normal_direct'}}
@property(nonatomic, readonly, direct) int direct_direct; // expected-error {{direct property declaration conflicts with previous direct declaration of property 'direct_direct'}}
@end

@implementation Category_Category
@end

__attribute__((objc_root_class))
@interface Category_CategoryImplementation
@end

@interface Category_CategoryImplementation (SomeCategory)
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly, direct) int direct_normal;
@property(nonatomic, readonly) int normal_direct; // expected-note {{previous declaration is here}}
@property(nonatomic, readonly, direct) int direct_direct;
@end

@implementation Category_CategoryImplementation (SomeCategory)
- (int)normal_normal {
  return 42;
}
- (int)direct_normal {
  return 42;
}
- (int)normal_direct __attribute__((objc_direct)) { // expected-error {{direct method implementation was previously declared not direct}}
  return 42;
}
- (int)direct_direct __attribute__((objc_direct)) {
  return 42;
}
@end

@implementation Category_CategoryImplementation
@end

__attribute__((objc_root_class))
@interface Interface_CategoryImplementation
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly, direct) int direct_normal; // expected-note {{previous declaration is here}}
@property(nonatomic, readonly) int normal_direct;         // expected-note {{previous declaration is here}}
@property(nonatomic, readonly, direct) int direct_direct; // expected-note {{previous declaration is here}}
@end

@interface Interface_CategoryImplementation (SomeCategory)
@end

@implementation Interface_CategoryImplementation (SomeCategory)
- (int)normal_normal {
  return 42;
}
- (int)direct_normal { // expected-error {{direct method was declared in the primary interface but is implemented in a category}}
  return 42;
}
- (int)normal_direct __attribute__((objc_direct)) { // expected-error {{direct method was declared in the primary interface but is implemented in a category}}
  return 42;
}
- (int)direct_direct __attribute__((objc_direct)) { // expected-error {{direct method was declared in the primary interface but is implemented in a category}}
  return 42;
}
@end

@implementation Interface_CategoryImplementation
@end

__attribute__((objc_root_class))
@interface Extension_CategoryImplementation
@end

@interface Extension_CategoryImplementation ()
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly, direct) int direct_normal; // expected-note {{previous declaration is here}}
@property(nonatomic, readonly) int normal_direct;         // expected-note {{previous declaration is here}}
@property(nonatomic, readonly, direct) int direct_direct; // expected-note {{previous declaration is here}}
@end

@interface Extension_CategoryImplementation (SomeCategory)
@end

@implementation Extension_CategoryImplementation (SomeCategory)
- (int)normal_normal {
  return 42;
}
- (int)direct_normal { // expected-error {{direct method was declared in an extension but is implemented in a different category}}
  return 42;
}
- (int)normal_direct __attribute__((objc_direct)) { // expected-error {{direct method was declared in an extension but is implemented in a different category}}
  return 42;
}
- (int)direct_direct __attribute__((objc_direct)) { // expected-error {{direct method was declared in an extension but is implemented in a different category}}
  return 42;
}
@end

__attribute__((objc_root_class))
@interface OtherCategory_CategoryImplementation
@end

@interface OtherCategory_CategoryImplementation (SomeCategory)
@end

@interface OtherCategory_CategoryImplementation (SomeOtherCategory)
@property(nonatomic, readonly) int normal_normal;
@property(nonatomic, readonly, direct) int direct_normal; // expected-note {{previous declaration is here}}
@property(nonatomic, readonly) int normal_direct;         // expected-note {{previous declaration is here}}
@property(nonatomic, readonly, direct) int direct_direct; // expected-note {{previous declaration is here}}
@end

@implementation OtherCategory_CategoryImplementation (SomeCategory)
- (int)normal_normal {
  return 42;
}
- (int)direct_normal { // expected-error {{direct method was declared in a category but is implemented in a different category}}
  return 42;
}
- (int)normal_direct __attribute__((objc_direct)) { // expected-error {{direct method was declared in a category but is implemented in a different category}}
  return 42;
}
- (int)direct_direct __attribute__((objc_direct)) { // expected-error {{direct method was declared in a category but is implemented in a different category}}
  return 42;
}
@end

@implementation OtherCategory_CategoryImplementation
@end
