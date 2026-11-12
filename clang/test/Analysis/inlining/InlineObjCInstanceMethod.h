
// Define a public header for the ObjC methods that are "visible" externally
// and, thus, could be sub-classed. We should explore the path on which these
// are sub-classed with unknown class by not inlining them.

typedef signed char BOOL;
typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
+(id)new;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end

@interface PublicClass : NSObject {
  int value3;
}
- (int)getZeroPublic;

- (int) value2;

@property (readonly) int value1;

@property int value3;
- (int)value3;
- (void)setValue3:(int)newValue;
@end

@interface PublicSubClass : PublicClass
@end

@interface PublicParent : NSObject
- (int)getZeroOverridden;
@end

@interface PublicSubClass2 : PublicParent
- (int) getZeroOverridden;
@end

