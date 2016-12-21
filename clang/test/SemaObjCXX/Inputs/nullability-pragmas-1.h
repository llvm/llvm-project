__attribute__((objc_root_class))
@interface NSError
@end

__attribute__((objc_root_class))
@interface A
@end

struct X { };

void f1(int *x); // expected-warning{{pointer is missing a nullability type specifier}}

typedef struct __attribute__((objc_bridge(NSError))) __CFError *CFErrorRef;
typedef NSError *NSErrorPtr;
typedef NSError **NSErrorPtrPtr;
typedef CFErrorRef *CFErrorRefPtr;
typedef int *int_ptr;
typedef A *A_ptr;
typedef int (^block_ptr)(int, int);

#pragma clang assume_nonnull begin

void f2(int *x);
void f3(A* obj);
void f4(int (^block)(int, int));
void f5(int_ptr x);
void f6(A_ptr obj);
void f7(int * _Nullable x);
void f8(A * _Nullable obj);
void f9(int X::* mem_ptr);
void f10(int (X::*mem_func)(int, int));
void f11(int X::* _Nullable mem_ptr);
void f12(int (X::* _Nullable mem_func)(int, int));

int_ptr f13(void);
A *f14(void);

int * _Null_unspecified f15(void);
A * _Null_unspecified f16(void);
void f17(CFErrorRef *error); // expected-note{{no known conversion from 'A * _Nonnull' to 'CFErrorRef  _Nullable * _Nullable' (aka '__CFError **') for 1st argument}}
void f18(A **); // expected-warning 2{{pointer is missing a nullability type specifier}}
void f19(CFErrorRefPtr error); // expected-warning{{pointer is missing a nullability type specifier}}

void g1(int (^)(int, int));
void g2(int (^ *bp)(int, int)); // expected-warning{{block pointer is missing a nullability type specifier}}
// expected-warning@-1{{pointer is missing a nullability type specifier}}
void g3(block_ptr *bp); // expected-warning{{block pointer is missing a nullability type specifier}}
// expected-warning@-1{{pointer is missing a nullability type specifier}}
void g4(int (*fp)(int, int));
void g5(int (**fp)(int, int)); // expected-warning 2{{pointer is missing a nullability type specifier}}

@interface A(Pragmas1)
+ (instancetype)aWithA:(A *)a;
- (A *)method1:(A_ptr)ptr;
- (null_unspecified A *)method2;
- (void)method3:(NSError **)error; // expected-note{{passing argument to parameter 'error' here}}
- (void)method4:(NSErrorPtr *)error; // expected-note{{passing argument to parameter 'error' here}}
- (void)method5:(NSErrorPtrPtr)error;
// expected-warning@-1{{pointer is missing a nullability type specifier}}

@property A *aProp;
@property NSError **anError; // expected-warning 2{{pointer is missing a nullability type specifier}}
@end

int *global_int_ptr;

// typedefs not inferred _Nonnull
typedef int *int_ptr_2;

typedef int * // expected-warning{{pointer is missing a nullability type specifier}}
            *int_ptr_ptr;

static inline void f30(void) {
  float *fp = global_int_ptr; // expected-error{{cannot initialize a variable of type 'float *' with an lvalue of type 'int * _Nonnull'}}

  int_ptr_2 ip2;
  float *fp2 = ip2; // expected-error{{cannot initialize a variable of type 'float *' with an lvalue of type 'int_ptr_2' (aka 'int *')}}

  int_ptr_ptr ipp;
  float *fp3 = ipp; // expected-error{{lvalue of type 'int_ptr_ptr' (aka 'int **')}}
}

@interface AA : A {
@public
  id ivar1;
  _Nonnull id ivar2;
}
@end

#pragma clang assume_nonnull end

void f20(A *a); // expected-warning{{pointer is missing a nullability type specifier}}
void f21(int_ptr x); // expected-warning{{pointer is missing a nullability type specifier}}
void f22(A_ptr y); // expected-warning{{pointer is missing a nullability type specifier}}
void f23(int_ptr _Nullable x);
void f24(A_ptr _Nullable y);
void f25(int_ptr_2 x); // expected-warning{{pointer is missing a nullability type specifier}}

@interface A(OutsidePragmas1)
+ (instancetype)aWithInt:(int)value; // expected-warning{{pointer is missing a nullability type specifier}}
@end
