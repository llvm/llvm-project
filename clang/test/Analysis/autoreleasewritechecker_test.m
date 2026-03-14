// UNSUPPORTED: system-windows
// RUN: %clang_analyze_cc1 -DARC -fobjc-arc -analyzer-checker=core,osx.cocoa.AutoreleaseWrite %s -triple x86_64-darwin -fblocks -verify
// RUN: %clang_analyze_cc1 -DNOARC -analyzer-checker=core,osx.cocoa.AutoreleaseWrite %s -fblocks -triple x86_64-darwin -verify


typedef signed char BOOL;
#define YES ((BOOL)1)
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
typedef unsigned long NSUInteger;

@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSError : NSObject <NSCopying, NSCoding> {}
+ (id)errorWithDomain:(int)domain;
@end

typedef int dispatch_semaphore_t;
typedef void (^block_t)();

typedef enum {
  NSEnumerationConcurrent = (1UL << 0),
  NSEnumerationReverse = (1UL << 1)
} NSEnumerationOptions;

@interface NSArray
- (void)enumerateObjectsUsingBlock:(block_t)block;
@end

@interface NSSet
- (void)objectsPassingTest:(block_t)block;
@end

@interface NSDictionary
- (void)enumerateKeysAndObjectsUsingBlock:(block_t)block;
@end

@interface NSIndexSet
- (void)indexesPassingTest:(block_t)block;
- (NSUInteger)indexWithOptions:(NSEnumerationOptions)opts
                   passingTest:(BOOL (^)(NSUInteger idx, BOOL *stop))predicate;
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
- (BOOL)writeToErrorInAutoreleasePool:(NSError *__autoreleasing *)error;
- (BOOL)writeToStrongErrorInAutoreleasePool:(NSError *__strong *)error;
- (BOOL)writeToLocalErrorInAutoreleasePool:(NSError *__autoreleasing *)error;
- (BOOL)writeToErrorInAutoreleasePoolMultipleTimes:(NSError *__autoreleasing *)error;
@end

@implementation I

- (BOOL) writeToErrorInBlock:(NSError *__autoreleasing *)error {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0l);
    dispatch_async(queue, ^{
        if (error) {
            *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside autorelease pool that may exit before method returns; consider writing first to a strong local variable declared outside of the block}}
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
            *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside autorelease pool that may exit before method returns; consider writing first to a strong local variable declared outside of the block}}
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
            *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside autorelease pool that may exit before method returns; consider writing first to a strong local variable declared outside of the block}}
        }
        dispatch_semaphore_signal(sem);
    });
    dispatch_async(queue, ^{
        if (error) {
            *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside autorelease pool that may exit before method returns; consider writing first to a strong local variable declared outside of the block}}
            *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside autorelease pool that may exit before method returns; consider writing first to a strong local variable declared outside of the block}}
        }
        dispatch_semaphore_signal(sem);
    });
    *error = [NSError errorWithDomain:1]; // no-warning

    dispatch_semaphore_wait(sem, 100);
    return 0;
}

- (BOOL)writeToErrorInAutoreleasePool:(NSError *__autoreleasing *)error {
  @autoreleasepool {
    if (error) {
      *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside locally-scoped autorelease pool; consider writing first to a strong local variable declared outside of the autorelease pool}}
    }
  }

  return 0;
}

- (BOOL)writeToStrongErrorInAutoreleasePool:(NSError *__strong *)error {
  @autoreleasepool {
    if (error) {
      *error = [NSError errorWithDomain:1]; // no-warning
    }
  }

  return 0;
}

- (BOOL)writeToLocalErrorInAutoreleasePool:(NSError *__autoreleasing *)error {
  NSError *localError;
  @autoreleasepool {
    localError = [NSError errorWithDomain:1]; // no-warning
  }

  if (error) {
    *error = localError; // no-warning
  }

  return 0;
}

- (BOOL)writeToErrorInAutoreleasePoolMultipleTimes:(NSError *__autoreleasing *)error {
  @autoreleasepool {
    if (error) {
      *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside locally-scoped autorelease pool; consider writing first to a strong local variable declared outside of the autorelease pool}}
    }
  }
  if (error) {
    *error = [NSError errorWithDomain:1]; // no-warning
  }
  @autoreleasepool {
    if (error) {
      *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside locally-scoped autorelease pool; consider writing first to a strong local variable declared outside of the autorelease pool}}
      *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside locally-scoped autorelease pool; consider writing first to a strong local variable declared outside of the autorelease pool}}
    }
  }

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
            *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside autorelease pool that may exit before function returns; consider writing first to a strong local variable declared outside of the block}}
        }
        dispatch_semaphore_signal(sem);
    });

    dispatch_semaphore_wait(sem, 100);
  return 0;
}

BOOL writeIntoErrorInAutoreleasePoolFromCFunc(NSError *__autoreleasing *error) {
  @autoreleasepool {
    if (error) {
      *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside locally-scoped autorelease pool; consider writing first to a strong local variable declared outside of the autorelease pool}}
    }
  }
  return 0;
}

BOOL writeToErrorNoWarning(NSError *__autoreleasing* error) {
  *error = [NSError errorWithDomain:1]; // no-warning
  return 0;
}

BOOL writeToErrorWithIterator(NSError *__autoreleasing* error, NSArray *a, NSSet *s, NSDictionary *d, NSIndexSet *i) { [a enumerateObjectsUsingBlock:^{
    *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside autorelease pool that may exit before function returns; consider writing first to a strong local variable declared outside of the block}}
    }];
  [d enumerateKeysAndObjectsUsingBlock:^{
    *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside autorelease pool that may exit before function returns; consider writing first to a strong local variable declared outside of the block}}
    }];
  [s objectsPassingTest:^{
    *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside autorelease pool that may exit before function returns; consider writing first to a strong local variable declared outside of the block}}
    }];
  [i indexesPassingTest:^{
    *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside autorelease pool that may exit before function returns; consider writing first to a strong local variable declared outside of the block}}
    }];
  [i indexWithOptions: NSEnumerationReverse passingTest:^(NSUInteger idx, BOOL *stop) {
    *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter inside autorelease pool that may exit before function returns; consider writing first to a strong local variable declared outside of the block}}
    return YES;
    }];
  return 0;
}

void writeIntoError(NSError **error) {
  *error = [NSError errorWithDomain:1];
}

extern void readError(NSError *error);

void writeToErrorWithIteratorNonnull(NSError *__autoreleasing* _Nonnull error, NSDictionary *a) {
  [a enumerateKeysAndObjectsUsingBlock:^{
     *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter}}
  }];
}


void escapeErrorFromIterator(NSError *__autoreleasing* _Nonnull error, NSDictionary *a) {
  [a enumerateKeysAndObjectsUsingBlock:^{
     writeIntoError(error); // expected-warning{{Capture of autoreleasing out parameter}}
  }];
}

void noWarningOnRead(NSError *__autoreleasing* error, NSDictionary *a) {
  [a enumerateKeysAndObjectsUsingBlock:^{
     NSError* local = *error; // no-warning
  }];
}

void noWarningOnEscapeRead(NSError *__autoreleasing* error, NSDictionary *a) {
  [a enumerateKeysAndObjectsUsingBlock:^{
     readError(*error); // no-warning
  }];
}

@interface ErrorCapture
- (void) captureErrorOut:(NSError**) error;
- (void) captureError:(NSError*) error;
@end

void escapeErrorFromIteratorMethod(NSError *__autoreleasing* _Nonnull error,
                                   NSDictionary *a,
                                   ErrorCapture *capturer) {
  [a enumerateKeysAndObjectsUsingBlock:^{
      [capturer captureErrorOut:error]; // expected-warning{{Capture of autoreleasing out parameter}}
  }];
}

void noWarningOnEscapeReadMethod(NSError *__autoreleasing* error,
                                 NSDictionary *a,
                                 ErrorCapture *capturer) {
  [a enumerateKeysAndObjectsUsingBlock:^{
    [capturer captureError:*error]; // no-warning
  }];
}

void multipleErrors(NSError *__autoreleasing* error, NSDictionary *a) {
  [a enumerateKeysAndObjectsUsingBlock:^{
     writeIntoError(error); // expected-warning{{Capture of autoreleasing out parameter}}
     *error = [NSError errorWithDomain:1]; // expected-warning{{Write to autoreleasing out parameter}}
     writeIntoError(error); // expected-warning{{Capture of autoreleasing out parameter}}
  }];
}

typedef void (^errBlock)(NSError *__autoreleasing *error);

extern void expectError(errBlock);

void captureAutoreleasingVarFromBlock(NSDictionary *dict) {
  expectError(^(NSError *__autoreleasing *err) {
    [dict enumerateKeysAndObjectsUsingBlock:^{
      writeIntoError(err); // expected-warning{{Capture of autoreleasing out parameter 'err'}}
    }];
  });
}

#endif

