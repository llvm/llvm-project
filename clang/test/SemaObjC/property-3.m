// RUN: %clang_cc1 -verify %s

@interface I 
{
	id d1;
}
@property (readwrite, copy) id d1;
@property (readwrite, copy) id d2;
@end

@interface NOW : I
@property (readonly) id d1; // expected-warning {{attribute 'readonly' of property 'd1' restricts attribute 'readwrite' of property inherited from 'I'}} expected-warning {{property 'd1' 'copy' attribute does not match the property inherited from 'I'}}
@property (readwrite, copy) I* d2;
@end
