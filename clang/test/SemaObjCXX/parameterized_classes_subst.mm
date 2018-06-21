// RUN: %clang_cc1 -fblocks -fsyntax-only -std=c++11 %s -verify
//
// Test the substitution of type arguments for type parameters when
// using parameterized classes in Objective-C.

__attribute__((objc_root_class))
@interface NSObject
+ (instancetype)alloc;
- (instancetype)init;
@end

@protocol NSCopying
@end

@interface NSString : NSObject <NSCopying>
@end

@interface NSMutableString : NSString
@end

@interface NSNumber : NSObject <NSCopying>
@end

@interface NSArray<T> : NSObject <NSCopying> {
@public
  T *data; // don't try this at home
}
- (T)objectAtIndexedSubscript:(int)index;
+ (NSArray<T> *)array;
@property (copy,nonatomic) T lastObject;
@end

@interface NSMutableArray<T> : NSArray<T>
-(instancetype)initWithArray:(NSArray<T> *)array; // expected-note{{passing argument}}
- (void)setObject:(T)object atIndexedSubscript:(int)index; // expected-note 2{{passing argument to parameter 'object' here}}
@end

@interface NSStringArray : NSArray<NSString *>
@end

@interface NSSet<T> : NSObject <NSCopying>
- (T)firstObject;
@property (nonatomic, copy) NSArray<T> *allObjects;
@end

// Parameterized inheritance (simple case)
@interface NSMutableSet<U : id<NSCopying>> : NSSet<U>
- (void)addObject:(U)object; // expected-note 7{{passing argument to parameter 'object' here}}
@end

@interface Widget : NSObject <NSCopying>
@end

// Non-parameterized class inheriting from a specialization of a
// parameterized class.
@interface WidgetSet : NSMutableSet<Widget *>
@end

// Parameterized inheritance with a more interesting transformation in
// the specialization.
@interface MutableSetOfArrays<T> : NSMutableSet<NSArray<T>*>
@end

// Inheriting from an unspecialized form of a parameterized type.
@interface UntypedMutableSet : NSMutableSet
@end

@interface Window : NSObject
@end

@interface NSDictionary<K, V> : NSObject <NSCopying>
- (V)objectForKeyedSubscript:(K)key; // expected-note 2{{parameter 'key'}}
@end

@interface NSMutableDictionary<K : id<NSCopying>, V> : NSDictionary<K, V> // expected-note 2{{type parameter 'K' declared here}} \
// expected-note 2{{'NSMutableDictionary' declared here}}
- (void)setObject:(V)object forKeyedSubscript:(K)key;
// expected-note@-1 {{parameter 'object' here}}
// expected-note@-2 {{parameter 'object' here}}
// expected-note@-3 {{parameter 'key' here}}
// expected-note@-4 {{parameter 'key' here}}

@property (strong) K someRandomKey;
@end

@interface WindowArray : NSArray<Window *>
@end

@interface NSSet<T> (Searching)
- (T)findObject:(T)object;
@end


// --------------------------------------------------------------------------
// Message sends.
// --------------------------------------------------------------------------
void test_message_send_result(
       NSSet<NSString *> *stringSet,
       NSMutableSet<NSString *> *mutStringSet,
       WidgetSet *widgetSet,
       UntypedMutableSet *untypedMutSet,
       MutableSetOfArrays<NSString *> *mutStringArraySet,
       NSSet *set,
       NSMutableSet *mutSet,
       MutableSetOfArrays *mutArraySet,
       NSArray<NSString *> *stringArray,
       void (^block)(void)) {
  int *ip;
  ip = [stringSet firstObject]; // expected-error{{from incompatible type 'NSString *'}}
  ip = [mutStringSet firstObject]; // expected-error{{from incompatible type 'NSString *'}}
  ip = [widgetSet firstObject]; // expected-error{{from incompatible type 'Widget *'}}
  ip = [untypedMutSet firstObject]; // expected-error{{from incompatible type 'id'}}
  ip = [mutStringArraySet firstObject]; // expected-error{{from incompatible type 'NSArray<NSString *> *'}}
  ip = [set firstObject]; // expected-error{{from incompatible type 'id'}}
  ip = [mutSet firstObject]; // expected-error{{from incompatible type 'id'}}
  ip = [mutArraySet firstObject]; // expected-error{{from incompatible type 'id'}}
  ip = [block firstObject]; // expected-error{{from incompatible type 'id'}}

  ip = [stringSet findObject:@"blah"]; // expected-error{{from incompatible type 'NSString *'}}

  // Class messages.
  ip = [NSSet<NSString *> alloc]; // expected-error{{from incompatible type 'NSSet<NSString *> *'}}
  ip = [NSSet alloc]; // expected-error{{from incompatible type 'NSSet *'}}
  ip = [MutableSetOfArrays<NSString *> alloc]; // expected-error{{from incompatible type 'MutableSetOfArrays<NSString *> *'}}
  ip = [MutableSetOfArrays alloc];  // expected-error{{from incompatible type 'MutableSetOfArrays *'}}
  ip = [NSArray<NSString *> array]; // expected-error{{from incompatible type 'NSArray<NSString *> *'}}
  ip = [NSArray<NSString *><NSCopying> array]; // expected-error{{from incompatible type 'NSArray<NSString *> *'}}

  ip = [[NSMutableArray<NSString *> alloc] init];  // expected-error{{from incompatible type 'NSMutableArray<NSString *> *'}}

  [[NSMutableArray alloc] initWithArray: stringArray]; // okay
  [[NSMutableArray<NSString *> alloc] initWithArray: stringArray]; // okay
  [[NSMutableArray<NSNumber *> alloc] initWithArray: stringArray]; // expected-error{{parameter of type 'NSArray<NSNumber *> *' with an lvalue of type 'NSArray<NSString *> *'}}
}

void test_message_send_param(
       NSMutableSet<NSString *> *mutStringSet,
       WidgetSet *widgetSet,
       UntypedMutableSet *untypedMutSet,
       MutableSetOfArrays<NSString *> *mutStringArraySet,
       NSMutableSet *mutSet,
       MutableSetOfArrays *mutArraySet,
       void (^block)(void)) {
  Window *window;

  [mutStringSet addObject: window]; // expected-error{{parameter of type 'NSString *'}}
  [widgetSet addObject: window]; // expected-error{{parameter of type 'Widget *'}}
  [untypedMutSet addObject: window]; // expected-error{{parameter of type 'id<NSCopying>'}}
  [mutStringArraySet addObject: window]; // expected-error{{parameter of type 'NSArray<NSString *> *'}}
  [mutSet addObject: window]; // expected-error{{parameter of type 'id<NSCopying>'}}
  [mutArraySet addObject: window]; // expected-error{{parameter of type 'id<NSCopying>'}}
  [block addObject: window]; // expected-error{{parameter of type 'id<NSCopying>'}}
}

// --------------------------------------------------------------------------
// Property accesses.
// --------------------------------------------------------------------------
void test_property_read(
       NSSet<NSString *> *stringSet,
       NSMutableSet<NSString *> *mutStringSet,
       WidgetSet *widgetSet,
       UntypedMutableSet *untypedMutSet,
       MutableSetOfArrays<NSString *> *mutStringArraySet,
       NSSet *set,
       NSMutableSet *mutSet,
       MutableSetOfArrays *mutArraySet,
       NSMutableDictionary *mutDict) {
  int *ip;
  ip = stringSet.allObjects; // expected-error{{from incompatible type 'NSArray<NSString *> *'}}
  ip = mutStringSet.allObjects; // expected-error{{from incompatible type 'NSArray<NSString *> *'}}
  ip = widgetSet.allObjects; // expected-error{{from incompatible type 'NSArray<Widget *> *'}}
  ip = untypedMutSet.allObjects; // expected-error{{from incompatible type 'NSArray *'}}
  ip = mutStringArraySet.allObjects; // expected-error{{from incompatible type 'NSArray<NSArray<NSString *> *> *'}}
  ip = set.allObjects; // expected-error{{from incompatible type 'NSArray *'}}
  ip = mutSet.allObjects; // expected-error{{from incompatible type 'NSArray *'}}
  ip = mutArraySet.allObjects; // expected-error{{from incompatible type 'NSArray *'}}

  ip = mutDict.someRandomKey; // expected-error{{from incompatible type '__kindof id<NSCopying>'}}
}

void test_property_write(
       NSMutableSet<NSString *> *mutStringSet,
       WidgetSet *widgetSet,
       UntypedMutableSet *untypedMutSet,
       MutableSetOfArrays<NSString *> *mutStringArraySet,
       NSMutableSet *mutSet,
       MutableSetOfArrays *mutArraySet,
       NSMutableDictionary *mutDict) {
  int *ip;

  mutStringSet.allObjects = ip; // expected-error{{to 'NSArray<NSString *> *'}}
  widgetSet.allObjects = ip; // expected-error{{to 'NSArray<Widget *> *'}}
  untypedMutSet.allObjects = ip; // expected-error{{to 'NSArray *'}}
  mutStringArraySet.allObjects = ip; // expected-error{{to 'NSArray<NSArray<NSString *> *> *'}}
  mutSet.allObjects = ip; // expected-error{{to 'NSArray *'}}
  mutArraySet.allObjects = ip; // expected-error{{to 'NSArray *'}}

  mutDict.someRandomKey = ip; // expected-error{{to 'id<NSCopying>'}}
}

// --------------------------------------------------------------------------
// Subscripting
// --------------------------------------------------------------------------
void test_subscripting(
       NSArray<NSString *> *stringArray,
       NSMutableArray<NSString *> *mutStringArray,
       NSArray *array,
       NSMutableArray *mutArray,
       NSDictionary<NSString *, Widget *> *stringWidgetDict,
       NSMutableDictionary<NSString *, Widget *> *mutStringWidgetDict,
       NSDictionary *dict,
       NSMutableDictionary *mutDict) {
  int *ip;
  NSString *string;
  Widget *widget;
  Window *window;

  ip = stringArray[0]; // expected-error{{from incompatible type 'NSString *'}}

  ip = mutStringArray[0]; // expected-error{{from incompatible type 'NSString *'}}
  mutStringArray[0] = ip; // expected-error{{parameter of type 'NSString *'}}

  ip = array[0]; // expected-error{{from incompatible type 'id'}}

  ip = mutArray[0]; // expected-error{{from incompatible type 'id'}}
  mutArray[0] = ip; // expected-error{{parameter of type 'id'}}

  ip = stringWidgetDict[string]; // expected-error{{from incompatible type 'Widget *'}}
  widget = stringWidgetDict[widget]; // expected-error{{parameter of type 'NSString *'}}

  ip = mutStringWidgetDict[string]; // expected-error{{from incompatible type 'Widget *'}}
  widget = mutStringWidgetDict[widget]; // expected-error{{parameter of type 'NSString *'}}
  mutStringWidgetDict[string] = ip; // expected-error{{parameter of type 'Widget *'}}
  mutStringWidgetDict[widget] = widget; // expected-error{{parameter of type 'NSString *'}}

  ip = dict[string]; // expected-error{{from incompatible type 'id'}}

  ip = mutDict[string]; // expected-error{{incompatible type 'id'}}
  mutDict[string] = ip; // expected-error{{parameter of type 'id'}}

  widget = mutDict[window];
  mutDict[window] = widget; // expected-error{{parameter of type 'id<NSCopying>'}}
}

// --------------------------------------------------------------------------
// Instance variable access.
// --------------------------------------------------------------------------
void test_instance_variable(NSArray<NSString *> *stringArray,
                            NSArray *array) {
  int *ip;

  ip = stringArray->data; // expected-error{{from incompatible type 'NSString **'}}
  ip = array->data; // expected-error{{from incompatible type 'id *'}}
}

@implementation WindowArray
- (void)testInstanceVariable {
  int *ip;

  ip = data; // expected-error{{from incompatible type 'Window **'}}
}
@end

// --------------------------------------------------------------------------
// Implicit conversions.
// --------------------------------------------------------------------------
void test_implicit_conversions(NSArray<NSString *> *stringArray,
                               NSArray<NSNumber *> *numberArray,
                               NSMutableArray<NSString *> *mutStringArray,
                               NSArray *array,
                               NSMutableArray *mutArray) {
  // Specialized -> unspecialized (same level)
  array = stringArray;

  // Unspecialized -> specialized (same level)
  stringArray = array;

  // Specialized -> specialized failure (same level).
  stringArray = numberArray; // expected-error{{assigning to 'NSArray<NSString *> *' from incompatible type 'NSArray<NSNumber *> *'}}

  // Specialized -> specialized (different levels).
  stringArray = mutStringArray;

  // Specialized -> specialized failure (different levels).
  numberArray = mutStringArray; // expected-error{{assigning to 'NSArray<NSNumber *> *' from incompatible type 'NSMutableArray<NSString *> *'}}

  // Unspecialized -> specialized (different levels).
  stringArray = mutArray;

  // Specialized -> unspecialized (different levels).
  array = mutStringArray;
}

@interface NSCovariant1<__covariant T>
@end

@interface NSContravariant1<__contravariant T>
@end

void test_variance(NSCovariant1<NSString *> *covariant1,
                   NSCovariant1<NSMutableString *> *covariant2,
                   NSCovariant1<NSString *(^)(void)> *covariant3,
                   NSCovariant1<NSMutableString *(^)(void)> *covariant4,
                   NSCovariant1<id> *covariant5,
                   NSCovariant1<id<NSCopying>> *covariant6,
                   NSContravariant1<NSString *> *contravariant1,
                   NSContravariant1<NSMutableString *> *contravariant2) {
  covariant1 = covariant2; // okay
  covariant2 = covariant1; // expected-warning{{incompatible pointer types assigning to 'NSCovariant1<NSMutableString *> *' from 'NSCovariant1<NSString *> *'}}

  covariant3 = covariant4; // okay
  covariant4 = covariant3; // expected-warning{{incompatible pointer types assigning to 'NSCovariant1<NSMutableString *(^)()> *' from 'NSCovariant1<NSString *(^)()> *'}}

  covariant5 = covariant1; // okay
  covariant1 = covariant5; // okay: id is promiscuous

  covariant5 = covariant3; // okay
  covariant3 = covariant5; // okay

  contravariant1 = contravariant2; // expected-warning{{incompatible pointer types assigning to 'NSContravariant1<NSString *> *' from 'NSContravariant1<NSMutableString *> *'}}
  contravariant2 = contravariant1; // okay
}

// --------------------------------------------------------------------------
// Ternary operator
// --------------------------------------------------------------------------
void test_ternary_operator(NSArray<NSString *> *stringArray,
                           NSArray<NSNumber *> *numberArray,
                           NSMutableArray<NSString *> *mutStringArray,
                           NSStringArray *stringArray2,
                           NSArray *array,
                           NSMutableArray *mutArray,
                           int cond) {
  int *ip;
  id object;

  ip = cond ? stringArray : mutStringArray; // expected-error{{from incompatible type 'NSArray<NSString *> *'}}
  ip = cond ? mutStringArray : stringArray; // expected-error{{from incompatible type 'NSArray<NSString *> *'}}

  ip = cond ? stringArray2 : mutStringArray; // expected-error{{from incompatible type 'NSArray<NSString *> *'}}
  ip = cond ? mutStringArray : stringArray2; // expected-error{{from incompatible type 'NSArray<NSString *> *'}}

  ip = cond ? stringArray : mutArray; // expected-error{{from incompatible type 'NSArray *'}}

  ip = cond ? stringArray2 : mutArray; // expected-error{{from incompatible type 'NSArray *'}}

  ip = cond ? mutArray : stringArray; // expected-error{{from incompatible type 'NSArray *'}}

  ip = cond ? mutArray : stringArray2; // expected-error{{from incompatible type 'NSArray *'}}

  object = cond ? stringArray : numberArray; // expected-warning{{incompatible operand types ('NSArray<NSString *> *' and 'NSArray<NSNumber *> *')}}
}

// --------------------------------------------------------------------------
// super
// --------------------------------------------------------------------------
@implementation NSStringArray
- (void)useSuperMethod {
  int *ip;
  ip = super.lastObject; // expected-error{{from incompatible type 'NSString *'}}
  ip = [super objectAtIndexedSubscript:0]; // expected-error{{from incompatible type 'NSString *'}}
}

+ (void)useSuperMethod {
  int *ip;
  ip = super.array; // expected-error{{from incompatible type 'NSArray<NSString *> *'}}
  ip = [super array]; // expected-error{{from incompatible type 'NSArray<NSString *> *'}}
}
@end

// --------------------------------------------------------------------------
// Template instantiation
// --------------------------------------------------------------------------
template<typename K, typename V>
struct NSMutableDictionaryOf {
  typedef NSMutableDictionary<K, V> *type; // expected-error{{type argument 'NSObject *' does not satisfy the bound ('id<NSCopying>') of type parameter 'K'}}
};

template<typename ...Args>
struct VariadicNSMutableDictionaryOf {
  typedef NSMutableDictionary<Args...> *type; // expected-error{{type argument 'NSObject *' does not satisfy the bound ('id<NSCopying>') of type parameter 'K'}}
  // expected-error@-1{{too many type arguments for class 'NSMutableDictionary' (have 3, expected 2)}}
  // expected-error@-2{{too few type arguments for class 'NSMutableDictionary' (have 1, expected 2)}}
};

void testInstantiation() {
  int *ip;

  typedef NSMutableDictionaryOf<NSString *, NSObject *>::type Dict1;
  Dict1 d1 = ip; // expected-error{{cannot initialize a variable of type 'Dict1' (aka 'NSMutableDictionary<NSString *,NSObject *> *')}}

  typedef NSMutableDictionaryOf<NSObject *, NSString *>::type Dict2; // expected-note{{in instantiation of template}}
}

void testVariadicInstantiation() {
  int *ip;

  typedef VariadicNSMutableDictionaryOf<NSString *, NSObject *>::type Dict1;
  Dict1 d1 = ip; // expected-error{{cannot initialize a variable of type 'Dict1' (aka 'NSMutableDictionary<NSString *,NSObject *> *')}}

  typedef VariadicNSMutableDictionaryOf<NSObject *, NSString *>::type Dict2; // expected-note{{in instantiation of template}}

  typedef VariadicNSMutableDictionaryOf<NSString *, NSObject *, NSObject *>::type Dict3; // expected-note{{in instantiation of template}}

  typedef VariadicNSMutableDictionaryOf<NSString *>::type Dict3; // expected-note{{in instantiation of template}}
}

// --------------------------------------------------------------------------
// Parameterized classes are not templates
// --------------------------------------------------------------------------
template<template<typename T, typename U> class TT>
struct AcceptsTemplateTemplate { };

typedef AcceptsTemplateTemplate<NSMutableDictionary> TemplateTemplateFail1; // expected-error{{template argument for template template parameter must be a class template or type alias template}}

template<typename T>
struct DependentTemplate {
  typedef typename T::template apply<NSString *, NSObject *> type; // expected-error{{'apply' following the 'template' keyword does not refer to a template}}
};

struct NSMutableDictionaryBuilder {
  typedef NSMutableDictionary apply;
};

typedef DependentTemplate<NSMutableDictionaryBuilder>::type DependentTemplateFail1; // expected-note{{in instantiation of template class}}

template<typename K, typename V>
struct NonDependentTemplate {
  typedef NSMutableDictionaryBuilder::template apply<NSString *, NSObject *> type; // expected-error{{'apply' following the 'template' keyword does not refer to a template}}
  // expected-error@-1{{expected member name or }}
};

// However, one can use an alias template to turn a parameterized
// class into a template.
template<typename K, typename V>
using NSMutableDictionaryAlias = NSMutableDictionary<K, V>;

typedef AcceptsTemplateTemplate<NSMutableDictionaryAlias> TemplateTemplateAlias1; // okay


