// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x -fblocks %s

// rdar://11231426
typedef signed char BOOL;

void y(BOOL (^foo)());

void x() {
    y(^{
        return __objc_yes;
    });
}

@protocol NSCopying
- copy;
@end

@interface NSObject
@end

@interface NSNumber : NSObject <NSCopying>
-copy;
@end

@interface NSNumber (NSNumberCreation)
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithUnsignedChar:(unsigned char)value;
+ (NSNumber *)numberWithShort:(short)value;
+ (NSNumber *)numberWithUnsignedShort:(unsigned short)value;
+ (NSNumber *)numberWithInt:(int)value;
+ (NSNumber *)numberWithUnsignedInt:(unsigned int)value;
+ (NSNumber *)numberWithLong:(long)value;
+ (NSNumber *)numberWithUnsignedLong:(unsigned long)value;
+ (NSNumber *)numberWithLongLong:(long long)value;
+ (NSNumber *)numberWithUnsignedLongLong:(unsigned long long)value;
+ (NSNumber *)numberWithFloat:(float)value;
+ (NSNumber *)numberWithDouble:(double)value;
+ (NSNumber *)numberWithBool:(BOOL)value;
@end

@interface NSArray : NSObject <NSCopying>
-copy;
@end

@interface NSArray (NSArrayCreation)
+ (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id<NSCopying> [])keys count:(unsigned long)cnt;
@end

@interface NSString
@end

template<typename T>
struct ConvertibleTo {
  operator T();
};

template<typename T>
struct ExplicitlyConvertibleTo {
  explicit operator T();
};

template<typename T>
class PrivateConvertibleTo {
private:
  operator T(); // expected-note{{declared private here}}
};

template<typename T> ConvertibleTo<T> makeConvertible();

struct X {
  ConvertibleTo<id> x;
  ConvertibleTo<id> get();
};

template<typename T> T test_numeric_instantiation() {
  return @-17.42;
}

template id test_numeric_instantiation();

void test_convertibility(ConvertibleTo<NSArray*> toArray,
                         ConvertibleTo<id> toId,
                         ConvertibleTo<int (^)(int)> toBlock,
                         ConvertibleTo<int> toInt,
                         ExplicitlyConvertibleTo<NSArray *> toArrayExplicit) {
  id array = @[ 
               toArray,
               toId,
               toBlock,
               toInt // expected-error{{collection element of type 'ConvertibleTo<int>' is not an Objective-C object}}
              ];
  id array2 = @[ toArrayExplicit ]; // expected-error{{collection element of type 'ExplicitlyConvertibleTo<NSArray *>' is not an Objective-C object}}

  id array3 = @[ 
                makeConvertible<id>(),
                               makeConvertible<id>, // expected-error{{collection element of type 'ConvertibleTo<id> ()' is not an Objective-C object}}
               ];

  X x;
  id array4 = @[ x.x ];
  id array5 = @[ x.get ]; // expected-error{{reference to non-static member function must be called}}
  id array6 = @[ PrivateConvertibleTo<NSArray*>() ]; // expected-error{{operator NSArray *' is a private member of 'PrivateConvertibleTo<NSArray *>'}}
}

template<typename T>
void test_array_literals(T t) {
  id arr = @[ @17, t ]; // expected-error{{collection element of type 'int' is not an Objective-C object}}
}

template void test_array_literals(id);
template void test_array_literals(NSArray*);
template void test_array_literals(int); // expected-note{{in instantiation of function template specialization 'test_array_literals<int>' requested here}}

template<typename T, typename U>
void test_dictionary_literals(T t, U u) {
  NSObject *object;
  id dict = @{ 
    @17 : t, // expected-error{{collection element of type 'int' is not an Objective-C object}}
    u : @42 // expected-error{{collection element of type 'int' is not an Objective-C object}}
  };

  id dict2 = @{ 
    object : @"object" // expected-error{{cannot initialize a parameter of type 'const id<NSCopying>' with an rvalue of type 'NSObject *'}}
  }; 
}

template void test_dictionary_literals(id, NSArray*);
template void test_dictionary_literals(NSArray*, id);
template void test_dictionary_literals(int, id); // expected-note{{in instantiation of function template specialization 'test_dictionary_literals<int, id>' requested here}}
template void test_dictionary_literals(id, int); // expected-note{{in instantiation of function template specialization 'test_dictionary_literals<id, int>' requested here}}

template<typename ...Args>
void test_bad_variadic_array_literal(Args ...args) {
  id arr1 = @[ args ]; // expected-error{{initializer contains unexpanded parameter pack 'args'}}
}

template<typename ...Args>
void test_variadic_array_literal(Args ...args) {
  id arr1 = @[ args... ]; // expected-error{{collection element of type 'int' is not an Objective-C object}}
}
template void test_variadic_array_literal(id);
template void test_variadic_array_literal(id, NSArray*);
template void test_variadic_array_literal(id, int, NSArray*); // expected-note{{in instantiation of function template specialization 'test_variadic_array_literal<id, int, NSArray *>' requested here}}

template<typename ...Args>
void test_bad_variadic_dictionary_literal(Args ...args) {
  id dict = @{ args : @17 }; // expected-error{{initializer contains unexpanded parameter pack 'args'}}
}

// Test array literal pack expansions. 
template<typename T, typename U>
struct pair {
  T first;
  U second;
};

template<typename T, typename ...Ts, typename ... Us>
void test_variadic_dictionary_expansion(T t, pair<Ts, Us>... key_values) {
  id dict = @{ 
    t : key_values.second ..., // expected-error{{collection element of type 'int' is not an Objective-C object}}
    key_values.first : key_values.second ..., // expected-error{{collection element of type 'float' is not an Objective-C object}}
    key_values.second : t ...
  };
}

template void test_variadic_dictionary_expansion(id, 
                                                 pair<NSNumber*, id>,
                                                 pair<id, ConvertibleTo<id>>);
template void test_variadic_dictionary_expansion(NSNumber *, // expected-note{{in instantiation of function template specialization}}
                                                 pair<NSNumber*, int>,
                                                 pair<id, ConvertibleTo<id>>);
template void test_variadic_dictionary_expansion(NSNumber *, // expected-note{{in instantiation of function template specialization}}
                                                 pair<NSNumber*, id>,
                                                 pair<float, ConvertibleTo<id>>);

// Test parsing 
struct key {
  static id value;
};

id key;
id value;

void test_dictionary_colon() {
  id dict = @{ key : value };
}

void testConstExpr() {
  constexpr NSString *t0 = @"abc";
  constexpr NSString *t1 = @("abc");
}
