// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin11 -fsyntax-only -std=c++11 -verify %s

@class NSArray;

@interface NSMutableDictionary
- (id)objectForKeyedSubscript:(id)key;
- (void)setObject:(id)object forKeyedSubscript:(id)key; // expected-note {{passing argument to parameter 'object' here}}
@end

template<typename T, typename U, typename O>
void test_dictionary_subscripts(T base, U key, O obj) {
  base[key] = obj; // expected-error {{expected method to write array element not found on object of type 'NSMutableDictionary *'}} \
                   // expected-error {{cannot initialize a parameter of type 'id' with an lvalue of type 'int'}}
  obj = base[key];  // expected-error {{expected method to read array element not found on object of type 'NSMutableDictionary *'}} \
                    // expected-error {{incompatible pointer to integer conversion assigning to 'int' from 'id'}}
     
}

template void test_dictionary_subscripts(NSMutableDictionary*, id, NSArray *ns);

template void test_dictionary_subscripts(NSMutableDictionary*, NSArray *ns, id);

template void test_dictionary_subscripts(NSMutableDictionary*, int, id); // expected-note {{in instantiation of function template specialization 'test_dictionary_subscripts<NSMutableDictionary *, int, id>' requested here}}

template void test_dictionary_subscripts(NSMutableDictionary*, id, int); // expected-note {{in instantiation of function template specialization 'test_dictionary_subscripts<NSMutableDictionary *, id, int>' requested here}}


@interface NSMutableArray
- (id)objectAtIndexedSubscript:(int)index;
- (void)setObject:(id)object atIndexedSubscript:(int)index;
@end

template<typename T, typename U, typename O>
void test_array_subscripts(T base, U index, O obj) {
  base[index] = obj; // expected-error {{indexing expression is invalid because subscript type 'double' is not an integral or Objective-C pointer type}}
  obj = base[index]; // expected-error {{indexing expression is invalid because subscript type 'double' is not an integral or Objective-C pointer type}}
}

template void  test_array_subscripts(NSMutableArray *, int, id);
template void  test_array_subscripts(NSMutableArray *, short, id);
enum E { e };

template void  test_array_subscripts(NSMutableArray *, E, id);

template void  test_array_subscripts(NSMutableArray *, double, id); // expected-note {{in instantiation of function template specialization 'test_array_subscripts<NSMutableArray *, double, id>' requested here}}

template<typename T>
struct ConvertibleTo {
  operator T();
};

template<typename T>
struct ExplicitlyConvertibleTo {
  explicit operator T();
};

template<typename T> ConvertibleTo<T> makeConvertible();

struct X {
  ConvertibleTo<id> x;
  ConvertibleTo<id> get();
};

NSMutableArray *test_array_convertibility(ConvertibleTo<NSMutableArray*> toArray,
                         ConvertibleTo<id> toId,
                         ConvertibleTo<int (^)(int)> toBlock,
                         ConvertibleTo<int> toInt,
                         ExplicitlyConvertibleTo<NSMutableArray *> toArrayExplicit) {
  id array;

  array[1] = toArray;

  array[4] = array[1];
 
  toArrayExplicit[2] = toId; // expected-error {{type 'ExplicitlyConvertibleTo<NSMutableArray *>' does not provide a subscript operator}}

  return array[toInt];
  
}

id test_dict_convertibility(ConvertibleTo<NSMutableDictionary*> toDict,
                         ConvertibleTo<id> toId,
                         ConvertibleTo<int (^)(int)> toBlock,
                         ConvertibleTo<int> toInt,
                         ExplicitlyConvertibleTo<NSMutableDictionary *> toDictExplicit) {


  NSMutableDictionary *Dict;
  id Id;
  Dict[toId] = toBlock;

  Dict[toBlock] = toBlock;

  Dict[toBlock] = Dict[toId] = Dict[toBlock];

  Id = toDictExplicit[toId] = Id; // expected-error {{no viable overloaded operator[] for type 'ExplicitlyConvertibleTo<NSMutableDictionary *>'}}

  return Dict[toBlock];
}


template<typename ...Args>
void test_bad_variadic_array_subscripting(Args ...args) {
  id arr1;
  arr1[3] = args; // expected-error {{expression contains unexpanded parameter pack 'args'}}
}

template<typename ...Args>
void test_variadic_array_subscripting(Args ...args) {
  id arr[] = {args[3]...}; // which means: {a[3], b[3], c[3]};
}

template void test_variadic_array_subscripting(id arg1, NSMutableArray* arg2, id arg3);

@class Key;

template<typename Index, typename ...Args>
void test_variadic_dictionary_subscripting(Index I, Args ...args) {
  id arr[] = {args[I]...}; // which means: {a[3], b[3], c[3]};
}

template void test_variadic_dictionary_subscripting(Key *key, id arg1, NSMutableDictionary* arg2, id arg3);

template<int N>
id get(NSMutableArray *array) {
 return array[N]; // array[N] should be a value- and instantiation-dependent ObjCSubscriptRefExpr
}

struct WeirdIndex {
   operator int(); // expected-note {{type conversion function declared here}}
   operator id(); // expected-note {{type conversion function declared here}}
};

id FUNC(WeirdIndex w) {
  NSMutableArray *array;
  return array[w]; // expected-error {{indexing expression is invalid because subscript type 'WeirdIndex' has multiple type conversion functions}}
}

