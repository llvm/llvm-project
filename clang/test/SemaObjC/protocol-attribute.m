// RUN: %clang_cc1 -fsyntax-only -verify %s

__attribute ((unavailable))
@protocol FwProto; // expected-note{{marked unavailable}}

Class <FwProto> cFw = 0;  // expected-error {{'FwProto' is unavailable}}


__attribute ((deprecated)) @protocol MyProto1 // expected-note 7 {{declared here}}
@end

@protocol Proto2  <MyProto1>  // expected-warning {{'MyProto1' is deprecated}}
+method2;
@end


@interface MyClass1 <MyProto1>  // expected-warning {{'MyProto1' is deprecated}}
{
  Class isa;
}
@end

@interface Derived : MyClass1 <MyProto1>  // expected-warning {{'MyProto1' is deprecated}}
{
	id <MyProto1> ivar;  // expected-warning {{'MyProto1' is deprecated}}
}
@end

@interface MyClass1 (Category) <MyProto1, Proto2>  // expected-warning {{'MyProto1' is deprecated}}
@end



Class <MyProto1> clsP1 = 0;  // expected-warning {{'MyProto1' is deprecated}}

@protocol FwProto @end // expected-note{{marked unavailable}}

@interface MyClass2 <FwProto> // expected-error {{'FwProto' is unavailable}}
@end

__attribute ((unavailable)) __attribute ((deprecated)) @protocol XProto; // expected-note{{marked unavailable}}

id <XProto> idX = 0; // expected-error {{'XProto' is unavailable}}

int main ()
{
	MyClass1 <MyProto1> *p1;  // expected-warning {{'MyProto1' is deprecated}}
}

