// RUN: %clang_cc1  -fsyntax-only -triple thumbv6-apple-ios3.0 -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -triple thumbv6-apple-ios3.0 -verify -Wno-objc-root-class %s
// rdar://12324295

typedef signed char BOOL;

@protocol P
@property(nonatomic,assign) id ptarget __attribute__((availability(ios,introduced=2.0,deprecated=3.0))); // expected-note 2 {{property 'ptarget' is declared deprecated here}}
@end

@protocol P1<P>
- (void)setPtarget:(id)arg; // expected-note {{method 'setPtarget:' declared here}}
@end


@interface UITableViewCell<P1>
@property(nonatomic,assign) id target __attribute__((availability(ios,introduced=2.0,deprecated=3.0))); // expected-note {{property 'target' is declared deprecated here}}
@end

@interface PSTableCell : UITableViewCell
 - (void)setTarget:(id)target; // expected-note {{method 'setTarget:' declared here}}
@end

@interface UITableViewCell(UIDeprecated)
@property(nonatomic,assign) id dep_target  __attribute__((availability(ios,introduced=2.0,deprecated=3.0))); // expected-note {{method 'dep_target' declared here}} \
                                                                                    // expected-note 2 {{property 'dep_target' is declared deprecated here}} \
                                                                                    // expected-note {{method 'setDep_target:' declared here}}
@end

@implementation PSTableCell
- (void)setTarget:(id)target {};
- (void)setPtarget:(id)val {};
- (void) Meth {
  [self setTarget: (id)0]; // expected-warning {{'setTarget:' is deprecated: first deprecated in iOS 3.0}}
  [self setDep_target: [self dep_target]]; // expected-warning {{'dep_target' is deprecated: first deprecated in iOS 3.0}} \
                                           // expected-warning {{'setDep_target:' is deprecated: first deprecated in iOS 3.0}}
					   
  [self setPtarget: (id)0]; // expected-warning {{setPtarget:' is deprecated: first deprecated in iOS 3.0}}
}
@end


@interface CustomAccessorNames
@property(getter=isEnabled,assign) BOOL enabled __attribute__((availability(ios,introduced=2.0,deprecated=3.0))); // expected-note {{method 'isEnabled' declared here}} expected-note {{property 'enabled' is declared deprecated here}}

@property(setter=setNewDelegate:,assign) id delegate __attribute__((availability(ios,introduced=2.0,deprecated=3.0))); // expected-note {{method 'setNewDelegate:' declared here}} expected-note {{property 'delegate' is declared deprecated here}}
@end

void testCustomAccessorNames(CustomAccessorNames *obj) {
  if ([obj isEnabled]) // expected-warning {{'isEnabled' is deprecated: first deprecated in iOS 3.0}}
    [obj setNewDelegate:0]; // expected-warning {{'setNewDelegate:' is deprecated: first deprecated in iOS 3.0}}
}


@interface ProtocolInCategory
@end

@interface ProtocolInCategory (TheCategory) <P1>
- (id)ptarget; // expected-note {{method 'ptarget' declared here}}
@end

id useDeprecatedProperty(ProtocolInCategory *obj) {
  return [obj ptarget]; // expected-warning {{'ptarget' is deprecated: first deprecated in iOS 3.0}}
}
