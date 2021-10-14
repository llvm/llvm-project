// RUN: %clang_cc1 -arcmt-action=check -verify -triple x86_64-apple-darwin10 -fsyntax-only -fblocks %s

// Classes that have an Objective-C object pointer.
struct HasObjectMember0 {
  id x;
};

struct HasObjectMember1 {
  id x[3];
};

struct HasObjectMember2 {
  id x[3][2];
};

// Don't complain if the type has non-external linkage
namespace {
  struct HasObjectMember3 {
    id x[3][2];
  };
}

// Don't complain if the Objective-C pointer type was explicitly given
// no lifetime.
struct HasObjectMember3 { 
  __unsafe_unretained id x[3][2];
};

struct HasBlockPointerMember0 {
  int (^bp)(int);
};

struct HasBlockPointerMember1 {
  int (^bp[2][3])(int);
};

struct NonPOD {
  NonPOD(const NonPOD&);
};

struct HasObjectMemberAndNonPOD0 {
  id x;
  NonPOD np;
};

struct HasObjectMemberAndNonPOD1 {
  NonPOD np;
  id x[3];
};

struct HasObjectMemberAndNonPOD2 {
  NonPOD np;
  id x[3][2];
};

struct HasObjectMemberAndNonPOD3 {
  HasObjectMemberAndNonPOD3 &operator=(const HasObjectMemberAndNonPOD3&);
  ~HasObjectMemberAndNonPOD3();
  NonPOD np;
  id x[3][2];
};

struct HasBlockPointerMemberAndNonPOD0 {
  NonPOD np;
  int (^bp)(int);
};

struct HasBlockPointerMemberAndNonPOD1 {
  NonPOD np;
  int (^bp[2][3])(int);
};

int check_non_pod_objc_pointer0[__is_pod(id)? 1 : -1];
int check_non_pod_objc_pointer1[__is_pod(__strong id)? -1 : 1];
int check_non_pod_objc_pointer2[__is_pod(__unsafe_unretained id)? 1 : -1];
int check_non_pod_objc_pointer3[__is_pod(id[2][3])? 1 : -1];
int check_non_pod_objc_pointer4[__is_pod(__unsafe_unretained id[2][3])? 1 : -1];
int check_non_pod_block0[__is_pod(int (^)(int))? 1 : -1];
int check_non_pod_block1[__is_pod(int (^ __unsafe_unretained)(int))? 1 : -1];

struct FlexibleArrayMember0 {
  int length;
  id array[]; // expected-error{{flexible array member 'array' of type '__strong id []' with non-trivial destruction}}
};

struct FlexibleArrayMember1 {
  int length;
  __unsafe_unretained id array[];
};

// It's okay to pass a retainable type through an ellipsis.
void variadic(...);
void test_variadic() {
  variadic(1, 17, @"Foo");
}

// It's okay to create a VLA of retainable types.
void vla(int n) {
  id vla[n];
}
