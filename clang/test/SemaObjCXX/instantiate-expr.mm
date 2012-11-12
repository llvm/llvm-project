// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface A {
@public
  int ivar;
}
@property int prop;
@end

typedef struct objc_object {
    Class isa;
} *id;

// Test instantiation of value-dependent ObjCIvarRefExpr,
// ObjCIsaRefExpr, and ObjCPropertyRefExpr nodes.
A *get_an_A(unsigned);
id get_an_id(unsigned);

template<unsigned N, typename T, typename U, typename V>
void f(U value, V value2) {
  get_an_A(N)->ivar = value; // expected-error{{assigning to 'int' from incompatible type 'int *'}}
  get_an_A(N).prop = value2; // expected-error{{assigning to 'int' from incompatible type 'double *'}}
  T c = get_an_id(N)->isa; // expected-error{{cannot initialize a variable of type 'int' with an lvalue of type 'Class'}} \
                           // expected-warning 5 {{direct access to objective-c's isa is deprecated in favor of object_setClass() and object_getClass()}}
}

template void f<6, Class>(int, int); // expected-note{{in instantiation of}}
template void f<7, Class>(int*, int); // expected-note{{in instantiation of}}
template void f<8, Class>(int, double*); // expected-note{{in instantiation of}}
template void f<9, int>(int, int); // expected-note{{in instantiation of}}

// Test instantiation of unresolved member reference expressions to an
// ivar reference.
template<typename T, typename U, typename V>
void f2(T ptr, U value, V value2) {
  ptr->ivar = value; // expected-error{{assigning to 'int' from incompatible type 'int *'}}
  ptr.prop = value2; // expected-error{{assigning to 'int' from incompatible type 'double *'}}
}

template void f2(A*, int, int);
template void f2(A*, int*, int); // expected-note{{instantiation of}}
template void f2(A*, int, double*); // expected-note{{instantiation of}}

// Test instantiation of unresolved member referfence expressions to
// an isa.
template<typename T, typename U>
void f3(U ptr) {
  T c = ptr->isa; // expected-error{{cannot initialize a variable of type 'int' with an lvalue of type 'Class'}} \
                  // expected-warning 2 {{direct access to objective-c's isa is deprecated in favor of object_setClass() and object_getClass()}}
}

template void f3<Class>(id); // expected-note{{in instantiation of}}
template void f3<int>(id); // expected-note{{instantiation of}}

// Implicit setter/getter
@interface B
- (int)foo;
- (void)setFoo:(int)value;
@end

template<typename T>
void f4(B *b, T value) {
  b.foo = value; // expected-error{{assigning to 'int' from incompatible type 'int *'}}
}

template void f4(B*, int);
template void f4(B*, int*); // expected-note{{in instantiation of function template specialization 'f4<int *>' requested here}}

template<typename T, typename U>
void f5(T ptr, U value) {
  ptr.foo = value; // expected-error{{assigning to 'int' from incompatible type 'int *'}}
}

template void f5(B*, int);
template void f5(B*, int*); // expected-note{{in instantiation of function template specialization 'f5<B *, int *>' requested here}}
