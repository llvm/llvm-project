#pragma clang system_header

#define nil ((id)0)

typedef signed char BOOL;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (Class)class;
@end

@interface NSObject <NSObject> {}
+ (instancetype)alloc;
- (void)dealloc;
- (id)init;
- (id)retain;
- (oneway void)release;
@end

@interface NSRunLoop : NSObject
+ (NSRunLoop *)currentRunLoop;
- (void)cancelPerformSelectorsWithTarget:(id)target;
@end

@interface NSNotificationCenter : NSObject
+ (NSNotificationCenter *)defaultCenter;
- (void)removeObserver:(id)observer;
@end

typedef struct objc_selector *SEL;
