// RUN: %clang_cc1 -fsyntax-only -verify -fobjc-exceptions %s

@interface NSException
@end

// @throw
template<typename T>
void throw_test(T value) {
  @throw value; // expected-error{{@throw requires an Objective-C object type ('int' invalid)}}
}

template void throw_test(NSException *);
template void throw_test(int); // expected-note{{in instantiation of}}

// @synchronized
template<typename T>
void synchronized_test(T value) {
  @synchronized (value) { // expected-error{{@synchronized requires an Objective-C object type ('int' invalid)}}
    value = 0;
  }
}

template void synchronized_test(NSException *);
template void synchronized_test(int); // expected-note{{in instantiation of}}

// fast enumeration
@interface NSArray
- (unsigned int)countByEnumeratingWithState:  (struct __objcFastEnumerationState *)state objects:  (id *)items count:(unsigned int)stackcount;
@end

@interface NSString
@end

struct vector {};

template<typename T> void eat(T);

template<typename E, typename T>
void fast_enumeration_test(T collection) {
  for (E element in collection) { // expected-error{{selector element type 'int' is not a valid object}} \
    // expected-error{{the type 'vector' is not a pointer to a fast-enumerable object}}
    eat(element);
  }

  E element;
  for (element in collection) // expected-error{{selector element type 'int' is not a valid object}} \
    // expected-error{{the type 'vector' is not a pointer to a fast-enumerable object}}
    eat(element);

  for (NSString *str in collection) // expected-error{{the type 'vector' is not a pointer to a fast-enumerable object}}
    eat(str);

  NSString *str;
  for (str in collection) // expected-error{{the type 'vector' is not a pointer to a fast-enumerable object}}
    eat(str);
}

template void fast_enumeration_test<NSString *>(NSArray*);
template void fast_enumeration_test<int>(NSArray*); // expected-note{{in instantiation of}}
template void fast_enumeration_test<NSString *>(vector); // expected-note{{in instantiation of}}

// @try/@catch/@finally

template<typename T, typename U>
void try_catch_finally_test(U value) {
  @try {
    value = 1; // expected-error{{assigning to 'int *' from incompatible type 'int'}}
  }
  @catch (T obj) { // expected-error{{@catch parameter is not a pointer to an interface type}}
    id x = obj;
  } @finally {
    value = 0;
  }
}

template void try_catch_finally_test<NSString *>(int);
template void try_catch_finally_test<NSString *>(int*); // expected-note{{in instantiation of}}
template void try_catch_finally_test<NSString>(int); // expected-note{{in instantiation of function template specialization 'try_catch_finally_test<NSString, int>' requested here}}
