// RUN: %clang_cc1 -fsyntax-only -verify %s  -fblocks

// Check property attribute consistency.

@interface I0
@property(readonly, readwrite) int p0; // expected-error {{property attributes 'readonly' and 'readwrite' are mutually exclusive}}

@property(retain) int p1; // expected-error {{property with 'retain (or strong)' attribute must be of object type}}
@property(strong) int s1; // expected-error {{property with 'retain (or strong)' attribute must be of object type}}

@property(copy) int p2; // expected-error {{property with 'copy' attribute must be of object type}}

@property(assign, copy) id p3_0; // expected-error {{property attributes 'assign' and 'copy' are mutually exclusive}} 
@property(assign, retain) id p3_1; // expected-error {{property attributes 'assign' and 'retain' are mutually exclusive}} 
@property(assign, strong) id s3_1; // expected-error {{property attributes 'assign' and 'strong' are mutually exclusive}} 
@property(copy, retain) id p3_2; // expected-error {{property attributes 'copy' and 'retain' are mutually exclusive}} 
@property(copy, strong) id s3_2; // expected-error {{property attributes 'copy' and 'strong' are mutually exclusive}} 
@property(assign, copy, retain) id p3_3; // expected-error {{property attributes 'assign' and 'copy' are mutually exclusive}}, expected-error {{property attributes 'assign' and 'retain' are mutually exclusive}} 
@property(assign, copy, strong) id s3_3; // expected-error {{property attributes 'assign' and 'copy' are mutually exclusive}}, expected-error {{property attributes 'assign' and 'strong' are mutually exclusive}} 

@property(unsafe_unretained, copy) id p4_0; // expected-error {{property attributes 'unsafe_unretained' and 'copy' are mutually exclusive}} 
@property(unsafe_unretained, retain) id p4_1; // expected-error {{property attributes 'unsafe_unretained' and 'retain' are mutually exclusive}} 
@property(unsafe_unretained, strong) id s4_1; // expected-error {{property attributes 'unsafe_unretained' and 'strong' are mutually exclusive}} 
@property(unsafe_unretained, copy, retain) id p4_3; // expected-error {{property attributes 'unsafe_unretained' and 'copy' are mutually exclusive}}, expected-error {{property attributes 'unsafe_unretained' and 'retain' are mutually exclusive}} 
@property(unsafe_unretained, copy, strong) id s4_3; // expected-error {{property attributes 'unsafe_unretained' and 'copy' are mutually exclusive}}, expected-error {{property attributes 'unsafe_unretained' and 'strong' are mutually exclusive}} 

@property id p4; // expected-warning {{no 'assign', 'retain', or 'copy' attribute is specified - 'assign' is assumed}}, expected-warning {{default property attribute 'assign' not appropriate for object}}

@property(nonatomic,copy) int (^includeMailboxCondition)(void); 
@property(nonatomic,copy) int (*includeMailboxCondition2)(void); // expected-error {{property with 'copy' attribute must be of object type}}

@end

@interface I0()
@property (retain) int PROP;	// expected-error {{property with 'retain (or strong)' attribute must be of object type}}
@property (strong) int SPROP;	// expected-error {{property with 'retain (or strong)' attribute must be of object type}}
@property(nonatomic,copy) int (*PROP1)(void); // expected-error {{property with 'copy' attribute must be of object type}}
@property(nonatomic,weak) int (*PROP2)(void); // expected-error {{property with 'weak' attribute must be of object type}}
@end

@interface rdar10357768
{
    int n1;
}
@property (readonly, setter=crushN1:) int n1; // expected-warning {{setter cannot be specified for a readonly property}}
@end

