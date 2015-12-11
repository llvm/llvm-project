// RUN: %clang_cc1 -verify %s

@interface I 
{
	id d1;
}
@property (readwrite, copy) id d1;
@property (readwrite, copy) id d2;
@end

@interface NOW : I
@property (readonly) id d1; // expected-warning {{attribute 'readonly' of property 'd1' restricts attribute 'readwrite' of property inherited from 'I'}} expected-warning {{'copy' attribute on property 'd1' does not match the property inherited from 'I'}}
@property (readwrite, copy) I* d2;
@end

// rdar://13156292
typedef signed char BOOL;

@protocol EKProtocolCalendar
@property (nonatomic, readonly) BOOL allowReminders;
@property (atomic, readonly) BOOL allowNonatomicProperty; // expected-note {{property declared here}}
@end

@protocol EKProtocolMutableCalendar <EKProtocolCalendar>
@end

@interface EKCalendar
@end

@interface EKCalendar ()  <EKProtocolMutableCalendar>
@property (nonatomic, assign) BOOL allowReminders;
@property (nonatomic, assign) BOOL allowNonatomicProperty; // expected-warning {{'atomic' attribute on property 'allowNonatomicProperty' does not match the property inherited from 'EKProtocolCalendar'}}
@end

__attribute__((objc_root_class))
@interface A
@property (nonatomic, readonly, getter=isAvailable) int available; // expected-note{{property declared here}}
@end

@interface A ()
@property (nonatomic, assign, getter=wasAvailable) int available; // expected-warning{{getter name mismatch between property redeclaration ('wasAvailable') and its original declaration ('isAvailable')}}
@end
