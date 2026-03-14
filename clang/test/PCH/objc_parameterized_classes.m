// RUN: %clang_cc1 -emit-pch %s -o %t
// RUN: %clang_cc1 -include-pch %t -verify %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED

@protocol NSObject
@end

__attribute__((objc_root_class))
@interface NSObject
@end

@interface PC1<__covariant T, U : NSObject *> : NSObject
// expected-note@-2{{type parameter 'U' declared here}}
@end

@interface PC1<__covariant T, U : NSObject *> (Cat1)
@end

typedef PC1<id, NSObject *> PC1Specialization1;

typedef PC1Specialization1 <NSObject> PC1Specialization2;
#else

@interface PC1<T : NSObject *, // expected-error{{type bound 'NSObject *' for type parameter 'T' conflicts with implicit bound 'id}}
 // expected-note@15{{type parameter 'T' declared here}}
               U : id> (Cat2) // expected-error{{type bound 'id' for type parameter 'U' conflicts with previous bound 'NSObject *'}}
 // expected-note@15{{type parameter 'U' declared here}}
@end

typedef PC1Specialization1<id, NSObject *> PC1Specialization3; // expected-error{{type arguments cannot be applied to already-specialized class type 'PC1Specialization1' (aka 'PC1<id,NSObject *>')}}

typedef PC1Specialization2<id, NSObject *> PC1Specialization4; // expected-error{{already-specialized class type 'PC1Specialization2' (aka 'PC1Specialization1<NSObject>')}}

@interface NSString : NSObject
@end

void testCovariance(PC1<NSObject *, NSObject *> *pc1a,
                    PC1<NSString *, NSObject *> *pc1b) {
  pc1a = pc1b;
}

#endif
