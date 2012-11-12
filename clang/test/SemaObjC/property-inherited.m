// RUN: %clang_cc1 %s -fsyntax-only -verify 
// RUN: %clang_cc1 -x objective-c++ %s -fsyntax-only -verify 

// rdar://6497242 Inherited overridden protocol declared objects don't work
// rdar://9740328 Case for c++

@protocol NSObject @end
@interface NSObject @end

@protocol FooDelegate<NSObject>
@optional
- (void)fooTask;
@end

@protocol BarDelegate<NSObject, FooDelegate>
@optional
- (void)barTask;
@end

@interface Foo : NSObject {
  id _delegate;
}
@property(nonatomic, assign) id<FooDelegate> delegate;
@property(nonatomic, assign) id<BarDelegate> delegate2; // expected-note {{property declared here}}
@end
@interface Bar : Foo {
}
@property(nonatomic, assign) id<BarDelegate> delegate;
@property(nonatomic, assign) id<FooDelegate> delegate2; // expected-warning{{property type 'id<FooDelegate>' is incompatible with type 'id<BarDelegate>' inherited from 'Foo'}}
@end

@interface NSData @end

@interface NSMutableData : NSData @end

@interface Base : NSData 
@property(assign) id ref;
@property(assign) Base *p_base;
@property(assign) NSMutableData *p_data;	 // expected-note {{property declared here}}
@end

@interface Data : Base 
@property(assign) NSData *ref;	
@property(assign) Data *p_base;	
@property(assign) NSData *p_data;	// expected-warning{{property type 'NSData *' is incompatible with type 'NSMutableData *' inherited from 'Base'}}
@end
