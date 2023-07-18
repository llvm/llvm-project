// RUN: rm -rf %t
// RUN: %clang_cc1 -fblocks -objcmt-migrate-readwrite-property -objcmt-migrate-readonly-property -mt-migrate-directory %t %s -x objective-c -fobjc-runtime-has-weak -fobjc-arc -triple x86_64-apple-darwin11
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c -fobjc-runtime-has-weak -fobjc-arc %s.result

#define WEBKIT_OBJC_METHOD_ANNOTATION(ANNOTATION) ANNOTATION
#define WEAK_IMPORT_ATTRIBUTE __attribute__((objc_arc_weak_reference_unavailable))
#define AVAILABLE_WEBKIT_VERSION_3_0_AND_LATER
#define DEPRECATED  __attribute__((deprecated)) 

typedef char BOOL;
@class NSString;
@protocol NSCopying @end

@interface NSObject <NSCopying>
@end

@interface NSDictionary : NSObject
@end

@interface I : NSObject {
  int ivarVal;
}
- (void) setWeakProp : (NSString *__weak)Val;
- (NSString *__weak) WeakProp;

- (NSString *) StrongProp;
- (void) setStrongProp : (NSString *)Val;

- (NSString *) UnavailProp  __attribute__((unavailable));
- (void) setUnavailProp  : (NSString *)Val;

- (NSString *) UnavailProp1  __attribute__((unavailable));
- (void) setUnavailProp1  : (NSString *)Val  __attribute__((unavailable));

- (NSString *) UnavailProp2;
- (void) setUnavailProp2  : (NSString *)Val  __attribute__((unavailable));

- (NSDictionary*) undoAction;
- (void) setUndoAction: (NSDictionary*)Arg;
@end

@implementation I
@end

@class NSArray;

@interface MyClass2  {
@private
    NSArray *_names1;
    NSArray *_names2;
    NSArray *_names3;
    NSArray *_names4;
}
- (void)setNames1:(NSArray *)names;
- (void)setNames4:(__strong NSArray *)names;
- (void)setNames3:(__strong NSArray *)names;
- (void)setNames2:(NSArray *)names;
- (NSArray *) names2;
- (NSArray *)names3;
- (__strong NSArray *)names4;
- (NSArray *) names1;
@end

// Properties that contain the name "delegate" or "dataSource",
// or have exact name "target" have unsafe_unretained attribute.
@interface NSInvocation 
- (id)target;
- (void)setTarget:(id)target;

- (id) dataSource;

// rdar://15509831
- (id)delegate;

- (id)xxxdelegateYYY;
- (void)setXxxdelegateYYY:(id)delegate;

- (void)setDataSource:(id)source;

- (id)MYtarget;
- (void)setMYtarget: (id)target;

- (id)targetX;
- (void)setTargetX: (id)t;
 
- (int)value;
- (void)setValue: (int)val;

-(BOOL) isContinuous;
-(void) setContinuous:(BOOL)value;

- (id) isAnObject;
- (void)setAnObject : (id) object;

- (BOOL) isinValid;
- (void) setInValid : (BOOL) arg;

- (void) Nothing;
- (int) Length;
- (id) object;
+ (double) D;
- (void *)JSObject WEBKIT_OBJC_METHOD_ANNOTATION(AVAILABLE_WEBKIT_VERSION_3_0_AND_LATER);
- (BOOL)isIgnoringInteractionEvents;

- (NSString *)getStringValue;
- (BOOL)getCounterValue;
- (void)setStringValue:(NSString *)stringValue AVAILABLE_WEBKIT_VERSION_3_0_AND_LATER;
- (NSDictionary *)getns_dixtionary;

- (BOOL)is3bar; // watch out
- (NSString *)get3foo; // watch out

- (BOOL) getM;
- (BOOL) getMA;
- (BOOL) getALL;
- (BOOL) getMANY;
- (BOOL) getSome;
@end


@interface NSInvocation(CAT)
- (id)target;
- (void)setTarget:(id)target;

- (id) dataSource;

- (id)xxxdelegateYYY;
- (void)setXxxdelegateYYY:(id)delegate;

- (void)setDataSource:(id)source;

- (id)MYtarget;
- (void)setMYtarget: (id)target;

- (id)targetX;
- (void)setTargetX: (id)t;

- (int)value;
- (void)setValue: (int)val;

-(BOOL) isContinuous;
-(void) setContinuous:(BOOL)value;

- (id) isAnObject;
- (void)setAnObject : (id) object;

- (BOOL) isinValid;
- (void) setInValid : (BOOL) arg;

- (void) Nothing;
- (int) Length;
- (id) object;
+ (double) D;

- (BOOL)is3bar; // watch out
- (NSString *)get3foo; // watch out

- (BOOL) getM;
- (BOOL) getMA;
- (BOOL) getALL;
- (BOOL) getMANY;
- (BOOL) getSome;
@end

DEPRECATED
@interface I_DEP
- (BOOL) isinValid;
- (void) setInValid : (BOOL) arg;
@end

@interface AnotherOne
- (BOOL) isinValid DEPRECATED;
- (void) setInValid : (BOOL) arg;
- (id)MYtarget;
- (void)setMYtarget: (id)target DEPRECATED;
- (BOOL) getM DEPRECATED;

- (id)xxxdelegateYYY DEPRECATED;
- (void)setXxxdelegateYYY:(id)delegate DEPRECATED;
@end

// rdar://14987909
#define NS_AVAILABLE __attribute__((availability(macosx,introduced=10.0)))
#define NORETURN __attribute__((noreturn))
#define ALIGNED __attribute__((aligned(16)))

@interface NSURL
// Do not infer a property.
- (NSURL *)appStoreReceiptURL NS_AVAILABLE;
- (void) setAppStoreReceiptURL : (NSURL *)object;

- (NSURL *)appStoreReceiptURLX NS_AVAILABLE;
- (void) setAppStoreReceiptURLX : (NSURL *)object NS_AVAILABLE;

// Do not infer a property.
- (NSURL *)appStoreReceiptURLY ;
- (void) setAppStoreReceiptURLY : (NSURL *)object NS_AVAILABLE;

- (id)OkToInfer NS_AVAILABLE;

// Do not infer a property.
- (NSURL *)appStoreReceiptURLZ ;
- (void) setAppStoreReceiptURLZ : (NSURL *)object NS_AVAILABLE;

// Do not infer a property.
- (id) t1 NORETURN NS_AVAILABLE;
- (void) setT1 : (id) arg NS_AVAILABLE;

- (id)method1 ALIGNED NS_AVAILABLE;
- (void) setMethod1 : (id) object NS_AVAILABLE ALIGNED;

- (NSURL *)init;  // No Change
+ (id)alloc;      // No Change

- (BOOL)is1stClass; // Not a valid property
- (BOOL)isClass;    // This is a valid property 'class' is not a keyword in ObjC
- (BOOL)isDouble; // Not a valid property

@end

// rdar://15082818
@class NSMutableDictionary;

@interface NSArray
- (id (^)(id, NSArray *, NSMutableDictionary *)) expressionBlock;
- (id (^)(id, NSArray *, NSMutableDictionary *)) MyBlock;
- (void) setMyBlock : (id (^)(id, NSArray *, NSMutableDictionary *)) bl;
- (id (*)(id, NSArray *, NSMutableDictionary *)) expressionFuncptr;
- (id (*)(id, NSArray *, NSMutableDictionary *)) MyFuncptr;
- (void) setMyFuncptr : (id (*)(id, NSArray *, NSMutableDictionary *)) bl;
@end

// rdar://15231241
@interface rdar15231241
@property (nonatomic, readonly) double Ddelegate;
@property (nonatomic, readonly) float Fdelegate;
@property (nonatomic, readonly) int Idelegate;
@property (nonatomic, readonly) BOOL Bdelegate;
@end

// rdar://19372798
@protocol NSObject @end
@protocol MyProtocol <NSObject>
- (id)readonlyProperty;
- (id)readWriteProperty;
- (void)setReadWriteProperty:(id)readWriteProperty;
@end
