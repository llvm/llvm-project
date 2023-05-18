// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o %t %s

// Superclass declares property. Subclass redeclares the same property.
// Do not @synthesize-by-default in the subclass. P1
// Superclass declares a property. Subclass declares a different property with the same name
// (such as different type or attributes). Do not @synthesize-by-default in the subclass. P2
// Superclass conforms to a protocol that declares a property. Subclass redeclares the 
// same property.  Do not @synthesize-by-default in the subclass. P3
// Superclass conforms to a protocol that declares a property. Subclass conforms to the 
// same protocol or a derived protocol. Do not @synthesize-by-default in the subclass. P4


@protocol PROTO
  @property int P3;
  @property int P4;
@end

@protocol PROTO1 <PROTO> 
  @property int IMP1;
@end

@interface Super <PROTO>
  @property int P1;
  @property (copy) id P2;
@end

@interface Sub : Super <PROTO1>
  @property int P1;
  @property (nonatomic, retain) id P2; // expected-warning {{property 'P2' 'copy' attribute does not match the property inherited from 'Super'}} \
				       // expected-warning {{property 'P2' 'atomic' attribute does not match the property inherited from 'Super'}}
  @property int P3;
  @property int IMP2;
@end

@implementation Sub 
@end

