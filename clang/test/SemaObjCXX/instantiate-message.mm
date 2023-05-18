// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test template instantiation of Objective-C message sends.

@interface ClassMethods
+ (ClassMethods *)method1:(void*)ptr;
@end

template<typename T>
struct identity {
  typedef T type;
};

template<typename R, typename T, typename Arg1>
void test_class_method(Arg1 arg1) {
  R *result1 = [T method1:arg1];
  R *result2 = [typename identity<T>::type method1:arg1];
  R *result3 = [ClassMethods method1:arg1]; // expected-error{{cannot initialize a variable of type 'ClassMethods2 *' with an rvalue of type 'ClassMethods *'}}
}

template void test_class_method<ClassMethods, ClassMethods>(void*);
template void test_class_method<ClassMethods, ClassMethods>(int*);

@interface ClassMethods2
+ (ClassMethods2 *)method1:(int*)ptr;
@end

template void test_class_method<ClassMethods2, ClassMethods2>(int*); // expected-note{{in instantiation of}}


@interface InstanceMethods
- (InstanceMethods *)method1:(void*)ptr;
@end

template<typename R, typename T, typename Arg1>
void test_instance_method(Arg1 arg1) {
  T *receiver = 0;
  InstanceMethods *im = 0;
  R *result1 = [receiver method1:arg1];
  R *result2 = [im method1:arg1]; // expected-error{{cannot initialize a variable of type 'InstanceMethods2 *' with an rvalue of type 'InstanceMethods *'}}
}

template void test_instance_method<InstanceMethods, InstanceMethods>(void*);
template void test_instance_method<InstanceMethods, InstanceMethods>(int*);

@interface InstanceMethods2
- (InstanceMethods2 *)method1:(void*)ptr;
@end

template void test_instance_method<InstanceMethods2, InstanceMethods2>(int*); // expected-note{{in instantiation of}}
