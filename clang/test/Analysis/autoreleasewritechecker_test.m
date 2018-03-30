// UNSUPPORTED: system-windows
// RUN: %clang_analyze_cc1 -DARC -fobjc-arc -analyzer-checker=core,osx.cocoa.AutoreleaseWrite %s -triple x86_64-darwin -fblocks -verify
// RUN: %clang_analyze_cc1 -DNOARC -analyzer-checker=core,osx.cocoa.AutoreleaseWrite %s -fblocks -triple x86_64-darwin -verify


typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
-(id)init;
-(id)autorelease;
-(id)copy;
-(id)retain;
@end
typedef int NSZone;
typedef int NSCoder;
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSError : NSObject <NSCopying, NSCoding> {}
+ (id)errorWithDomain:(int)domain;
@end

typedef int dispatch_semaphore_t;
typedef void (^block_t)();

@interface NSArray
- (void) enumerateObjectsUsingBlock:(block_t)block;
@end

typedef int group_t;
typedef struct dispatch_queue_s *dispatch_queue_t;
typedef void (^dispatch_block_t)(void);
extern dispatch_queue_t queue;

void dispatch_group_async(dispatch_queue_t queue,
                          group_t group,
                          dispatch_block_t block);
void dispatch_async(dispatch_queue_t queue, dispatch_block_t block);
dispatch_semaphore_t dispatch_semaphore_create(int);

void dispatch_semaphore_wait(dispatch_semaphore_t, int);
void dispatch_semaphore_signal(dispatch_semaphore_t);

// No warnings without ARC.
#ifdef NOARC

// expected-no-diagnostics
BOOL writeToErrorWithIterator(NSError ** error, NSArray *a) {
  [a enumerateObjectsUsingBlock:^{
    *error = [NSError errorWithDomain:1]; // no-warning
    }];
  return 0;
}
#endif

#ifdef ARC
@interface I : NSObject
- (BOOL) writeToStrongErrorInBlock:(NSError *__strong *)error;
- (BOOL) writeToErrorInBlock:(NSError *__autoreleasing *)error;
- (BOOL) writeToLocalErrorInBlock:(NSError **)error;
- (BOOL) writeToErrorInBlockMultipleTimes:(NSError *__autoreleasing *)error;
- (BOOL) writeToError:(NSError *__autoreleasing *)error;
- (BOOL) writeToErrorWithDispatchGroup:(NSError *__autoreleasing *)error;
@end

@implementation I

- (BOOL) writeToErrorInBlock:(NSError *__autoreleasing *)error {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0l);
    dispatch_async(queue, ^{
        if (error) {
            *error = [NSError errorWithDomain:1]; // expected-warning{{Writing into an auto-releasing out parameter inside autorelease pool that may exit before method returns}}
        }
        dispatch_semaphore_signal(sem);
    });

    dispatch_semaphore_wait(sem, 100);
    return 0;
}

- (BOOL) writeToErrorWithDispatchGroup:(NSError *__autoreleasing *)error {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0l);
    dispatch_group_async(queue, 0, ^{
        if (error) {
            *error = [NSError errorWithDomain:1]; // expected-warning{{Writing into an auto-releasing out}}
        }
        dispatch_semaphore_signal(sem);
    });

    dispatch_semaphore_wait(sem, 100);
    return 0;
}

- (BOOL) writeToLocalErrorInBlock:(NSError *__autoreleasing *)error {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0l);
    dispatch_async(queue, ^{
        NSError* error2;
        NSError*__strong* error3 = &error2;
        if (error) {
            *error3 = [NSError errorWithDomain:1]; // no-warning
        }
        dispatch_semaphore_signal(sem);
    });

    dispatch_semaphore_wait(sem, 100);
    return 0;
}

- (BOOL) writeToStrongErrorInBlock:(NSError *__strong *)error {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0l);
    dispatch_async(queue, ^{
        if (error) {
            *error = [NSError errorWithDomain:2]; // no-warning
        }
        dispatch_semaphore_signal(sem);
    });

    dispatch_semaphore_wait(sem, 100);
    return 0;
}

- (BOOL) writeToErrorInBlockMultipleTimes:(NSError *__autoreleasing *)error {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0l);
    dispatch_async(queue, ^{
        if (error) {
            *error = [NSError errorWithDomain:1]; // expected-warning{{Writing into an auto-releasing out}}
        }
        dispatch_semaphore_signal(sem);
    });
    dispatch_async(queue, ^{
        if (error) {
            *error = [NSError errorWithDomain:1]; // expected-warning{{Writing into an auto-releasing out}}
            *error = [NSError errorWithDomain:1]; // expected-warning{{Writing into an auto-releasing out}}
        }
        dispatch_semaphore_signal(sem);
    });
    *error = [NSError errorWithDomain:1]; // no-warning

    dispatch_semaphore_wait(sem, 100);
    return 0;
}

- (BOOL) writeToError:(NSError *__autoreleasing *)error {
    *error = [NSError errorWithDomain:1]; // no-warning
    return 0;
}
@end

BOOL writeToErrorInBlockFromCFunc(NSError *__autoreleasing* error) {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0l);
    dispatch_async(queue, ^{
        if (error) {
            *error = [NSError errorWithDomain:1]; // expected-warning{{Writing into an auto-releasing out}}
        }
        dispatch_semaphore_signal(sem);
    });

    dispatch_semaphore_wait(sem, 100);
  return 0;
}

BOOL writeToErrorNoWarning(NSError *__autoreleasing* error) {
  *error = [NSError errorWithDomain:1]; // no-warning
  return 0;
}

BOOL writeToErrorWithIterator(NSError *__autoreleasing* error, NSArray *a) {
  [a enumerateObjectsUsingBlock:^{
    *error = [NSError errorWithDomain:1]; // expected-warning{{Writing into an auto-releasing out parameter inside autorelease pool that may exit before function returns; consider writing first to a strong local variable declared outside of the block}}
    }];
  return 0;
}
#endif
