#if __has_feature(objc_arr)
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE __attribute__((unavailable("not available in automatic reference counting mode")))
#else
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE
#endif

#define NS_RETURNS_RETAINED __attribute__((ns_returns_retained))
#define CF_CONSUMED __attribute__((cf_consumed))
#define CF_RETURNS_RETAINED __attribute__((cf_returns_retained))

#define NS_INLINE static __inline__ __attribute__((always_inline))
#define nil ((void*) 0)

typedef int BOOL;
typedef unsigned NSUInteger;
typedef int int32_t;
typedef unsigned char uint8_t;
typedef int32_t UChar32;
typedef unsigned char UChar;

typedef struct _NSZone NSZone;

typedef const void * CFTypeRef;
CFTypeRef CFRetain(CFTypeRef cf);
CFTypeRef CFMakeCollectable(CFTypeRef cf) NS_AUTOMATED_REFCOUNT_UNAVAILABLE;

NS_INLINE NS_RETURNS_RETAINED id NSMakeCollectable(CFTypeRef CF_CONSUMED cf) NS_AUTOMATED_REFCOUNT_UNAVAILABLE;

@protocol NSObject
- (BOOL)isEqual:(id)object;
- (NSZone *)zone NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (id)retain NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (NSUInteger)retainCount NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (oneway void)release NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (id)autorelease NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
@end

@interface NSObject <NSObject> {}
- (id)init;

+ (id)new;
+ (id)alloc;
- (void)dealloc;

- (void)finalize;

- (id)copy;
- (id)mutableCopy;
@end

NS_AUTOMATED_REFCOUNT_UNAVAILABLE
@interface NSAutoreleasePool : NSObject {
@private
    void    *_token;
    void    *_reserved3;
    void    *_reserved2;
    void    *_reserved;
}

+ (void)addObject:(id)anObject;

- (void)addObject:(id)anObject;

- (void)drain;

@end

typedef const void* objc_objectptr_t; 
extern __attribute__((ns_returns_retained)) id objc_retainedObject(objc_objectptr_t __attribute__((cf_consumed)) pointer);
extern __attribute__((ns_returns_not_retained)) id objc_unretainedObject(objc_objectptr_t pointer);
extern objc_objectptr_t objc_unretainedPointer(id object);

#define dispatch_retain(object) ({ dispatch_object_t _o = (object); _dispatch_object_validate(_o); (void)[_o retain]; })
#define dispatch_release(object) ({ dispatch_object_t _o = (object); _dispatch_object_validate(_o); [_o release]; })
#define xpc_retain(object) ({ xpc_object_t _o = (object); _xpc_object_validate(_o); [_o retain]; })
#define xpc_release(object) ({ xpc_object_t _o = (object); _xpc_object_validate(_o); [_o release]; })

typedef id dispatch_object_t;
typedef id xpc_object_t;

void _dispatch_object_validate(dispatch_object_t object);
void _xpc_object_validate(xpc_object_t object);

#if __has_feature(objc_arc)

NS_INLINE CF_RETURNS_RETAINED CFTypeRef CFBridgingRetain(id X) {
    return (__bridge_retained CFTypeRef)X;
}

NS_INLINE id CFBridgingRelease(CFTypeRef CF_CONSUMED X) {
    return (__bridge_transfer id)X;
}

#else

NS_INLINE CF_RETURNS_RETAINED CFTypeRef CFBridgingRetain(id X) {
    return X ? CFRetain((CFTypeRef)X) : NULL;
}

NS_INLINE id CFBridgingRelease(CFTypeRef CF_CONSUMED X) {
    return [(id)CFMakeCollectable(X) autorelease];
}

#endif
