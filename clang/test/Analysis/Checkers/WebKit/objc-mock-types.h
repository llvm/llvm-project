@class NSString;
@class NSArray;
@class NSMutableArray;
@class NSDictionary;
@class NSMutableDictionary;
#define nil ((id)0)
#define CF_BRIDGED_TYPE(T) __attribute__((objc_bridge(T)))
#define CF_BRIDGED_MUTABLE_TYPE(T) __attribute__((objc_bridge_mutable(T)))
typedef CF_BRIDGED_TYPE(id) void * CFTypeRef;
typedef unsigned long long CFTypeID;
typedef signed char BOOL;
typedef unsigned char Boolean;
typedef signed long CFIndex;
typedef unsigned long NSUInteger;
typedef const struct CF_BRIDGED_TYPE(id) __CFAllocator * CFAllocatorRef;
typedef const struct CF_BRIDGED_TYPE(NSString) __CFString * CFStringRef;
typedef const struct CF_BRIDGED_TYPE(NSArray) __CFArray * CFArrayRef;
typedef struct CF_BRIDGED_MUTABLE_TYPE(NSMutableArray) __CFArray * CFMutableArrayRef;
typedef struct CF_BRIDGED_MUTABLE_TYPE(CFRunLoopRef) __CFRunLoop * CFRunLoopRef;
typedef struct CF_BRIDGED_TYPE(id) CGImage *CGImageRef;

#define NS_RETURNS_RETAINED __attribute__((ns_returns_retained))
#define CF_CONSUMED __attribute__((cf_consumed))
#define CF_RETURNS_RETAINED __attribute__((cf_returns_retained))
#define NS_REQUIRES_PROPERTY_DEFINITIONS __attribute__((objc_requires_property_definitions))

extern const CFAllocatorRef kCFAllocatorDefault;
typedef struct _NSZone NSZone;

CFTypeID CFGetTypeID(CFTypeRef cf);

CFTypeID CFGetTypeID(CFTypeRef cf);
CFTypeID CFArrayGetTypeID();
CFMutableArrayRef CFArrayCreateMutable(CFAllocatorRef allocator, CFIndex capacity);
extern void CFArrayAppendValue(CFMutableArrayRef theArray, const void *value);
CFArrayRef CFArrayCreate(CFAllocatorRef allocator, const void **values, CFIndex numValues);
CFIndex CFArrayGetCount(CFArrayRef theArray);

typedef const struct CF_BRIDGED_TYPE(NSDictionary) __CFDictionary * CFDictionaryRef;
typedef struct CF_BRIDGED_MUTABLE_TYPE(NSMutableDictionary) __CFDictionary * CFMutableDictionaryRef;

CFTypeID CFDictionaryGetTypeID();
CFDictionaryRef CFDictionaryCreate(CFAllocatorRef allocator, const void **keys, const void **values, CFIndex numValues);
CFDictionaryRef CFDictionaryCreateCopy(CFAllocatorRef allocator, CFDictionaryRef theDict);
CFDictionaryRef CFDictionaryCreateMutableCopy(CFAllocatorRef allocator, CFIndex capacity, CFDictionaryRef theDict);
CFIndex CFDictionaryGetCount(CFDictionaryRef theDict);
Boolean CFDictionaryContainsKey(CFDictionaryRef theDict, const void *key);
Boolean CFEqual(CFTypeRef, CFTypeRef);
Boolean CFDictionaryContainsValue(CFDictionaryRef theDict, const void *value);
const void *CFDictionaryGetValue(CFDictionaryRef theDict, const void *key);
void CFDictionaryAddValue(CFMutableDictionaryRef theDict, const void *key, const void *value);
void CFDictionarySetValue(CFMutableDictionaryRef theDict, const void *key, const void *value);
void CFDictionaryReplaceValue(CFMutableDictionaryRef theDict, const void *key, const void *value);
void CFDictionaryRemoveValue(CFMutableDictionaryRef theDict, const void *key);

typedef struct CF_BRIDGED_TYPE(id) __SecTask * SecTaskRef;
SecTaskRef SecTaskCreateFromSelf(CFAllocatorRef allocator);

typedef struct CF_BRIDGED_TYPE(id) CF_BRIDGED_MUTABLE_TYPE(IOSurface) __IOSurface * IOSurfaceRef;
IOSurfaceRef IOSurfaceCreate(CFDictionaryRef properties);

typedef struct CF_BRIDGED_TYPE(id) __CVBuffer *CVBufferRef;
typedef CVBufferRef CVImageBufferRef;
typedef CVImageBufferRef CVPixelBufferRef;
typedef signed int CVReturn;
CVReturn CVPixelBufferCreateWithIOSurface(CFAllocatorRef allocator, IOSurfaceRef surface, CFDictionaryRef pixelBufferAttributes, CF_RETURNS_RETAINED CVPixelBufferRef * pixelBufferOut);

CFRunLoopRef CFRunLoopGetCurrent(void);
CFRunLoopRef CFRunLoopGetMain(void);
extern CFTypeRef CFRetain(CFTypeRef cf);
extern void CFRelease(CFTypeRef cf);
#define CFSTR(cStr) ((CFStringRef) __builtin___CFStringMakeConstantString ("" cStr ""))
extern Class NSClassFromString(NSString *aClassName);

#if __has_feature(objc_arc)
id CFBridgingRelease(CFTypeRef X) {
    return (__bridge_transfer id)X;
}
#endif

__attribute__((objc_root_class))
@interface NSObject
+ (instancetype) alloc;
+ (Class) class;
+ (Class) superclass;
- (instancetype) init;
- (instancetype)retain;
- (instancetype)autorelease;
- (void)release;
- (BOOL)isKindOfClass:(Class)aClass;
@end

@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end

@protocol NSFastEnumeration
- (int)countByEnumeratingWithState:(void *)state objects:(id *)objects count:(unsigned)count;
- (void)protocolMethod;
@end

@interface NSEnumerator <NSFastEnumeration>
@end

@interface NSDictionary : NSObject <NSCopying>
- (NSUInteger)count;
- (id)objectForKey:(id)aKey;
- (id)objectForKeyedSubscript:(id)aKey;
- (NSEnumerator *)keyEnumerator;
+ (id)dictionary;
+ (id)dictionaryWithObject:(id)object forKey:(id <NSCopying>)key;
- (NSMutableDictionary *)mutableCopy;
+ (instancetype)dictionaryWithObjects:(const id [])objects forKeys:(const id <NSCopying> [])keys count:(NSUInteger)cnt;
@end

@interface NSArray : NSObject <NSCopying, NSFastEnumeration>
- (NSUInteger)count;
- (NSEnumerator *)objectEnumerator;
- (NSMutableArray *)mutableCopy;
+ (NSArray *)arrayWithObjects:(const id [])objects count:(NSUInteger)count;
@end

@interface NSString : NSObject <NSCopying>
- (NSUInteger)length;
- (NSString *)stringByAppendingString:(NSString *)aString;
- ( const char *)UTF8String;
- (id)initWithUTF8String:(const char *)nullTerminatedCString;
- (NSString *)copy;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end

@interface NSMutableString : NSString
@end

@interface NSValue : NSObject <NSCopying>
- (void)getValue:(void *)value;
@end

@interface NSNumber : NSValue
- (char)charValue;
- (int)intValue;
- (id)initWithInt:(int)value;
+ (NSNumber *)numberWithInt:(int)value;
@end

@interface SomeObj : NSObject
- (instancetype)_init;
- (SomeObj *)mutableCopy;
- (SomeObj *)copyWithValue:(int)value;
- (void)doWork;
- (SomeObj *)other;
- (SomeObj *)next;
- (int)value;
@end

@interface OtherObj : SomeObj
- (void)doMoreWork:(OtherObj *)other;
@end

namespace WTF {

void WTFCrash(void);

#if __has_feature(objc_arc)
#define RetainPtr RetainPtrArc
#endif

template<typename T> class RetainPtr;
template<typename T> RetainPtr<T> adoptNS(T*);
template<typename T> RetainPtr<T> adoptCF(T);

template <typename T, typename S> T *downcast(S *t) { return static_cast<T*>(t); }

template <typename T> struct RemovePointer {
  typedef T Type;
};

template <typename T> struct RemovePointer<T*> {
  typedef T Type;
};

template <typename T> struct RetainPtr {
  using ValueType = typename RemovePointer<T>::Type;
  using PtrType = ValueType*;
  PtrType t;

  RetainPtr() : t(nullptr) { }
  RetainPtr(PtrType t)
    : t(t) {
    if (t)
      CFRetain(toCFTypeRef(t));
  }
  RetainPtr(RetainPtr&&);
  RetainPtr(const RetainPtr&);
  template <typename U>
  RetainPtr(const RetainPtr<U>& o)
    : RetainPtr(o.t)
  {}
  RetainPtr operator=(const RetainPtr& o)
  {
    if (t)
      CFRelease(toCFTypeRef(t));
    t = o.t;
    if (t)
      CFRetain(toCFTypeRef(t));
    return *this;
  }
  template <typename U>
  RetainPtr operator=(const RetainPtr<U>& o)
  {
    if (t)
      CFRelease(toCFTypeRef(t));
    t = o.t;
    if (t)
      CFRetain(toCFTypeRef(t));
    return *this;
  }
  ~RetainPtr() {
    clear();
  }
  void clear() {
    if (t)
      CFRelease(toCFTypeRef(t));
    t = nullptr;
  }
  void swap(RetainPtr& o) {
    PtrType tmp = t;
    t = o.t;
    o.t = tmp;
  }
  PtrType get() const { return t; }
  PtrType operator->() const { return t; }
  T &operator*() const { return *t; }
  RetainPtr &operator=(PtrType t);
  PtrType leakRef()
  {
    PtrType s = t;
    t = nullptr;
    return s;
  }
  operator PtrType() const { return t; }
  operator bool() const { return t; }

#if !__has_feature(objc_arc)
  PtrType autorelease() { [[clang::suppress]] return [t autorelease]; }
#endif

private:
  CFTypeRef toCFTypeRef(id ptr) { return (__bridge CFTypeRef)ptr; }
  CFTypeRef toCFTypeRef(const void* ptr) { return (CFTypeRef)ptr; }

  template <typename U> friend RetainPtr<U> adoptNS(U*);
  template <typename U> friend RetainPtr<U> adoptCF(U);

  enum AdoptTag { Adopt };
  RetainPtr(PtrType t, AdoptTag) : t(t) { }
};

template <typename T>
RetainPtr<T>::RetainPtr(RetainPtr<T>&& o)
  : RetainPtr(o.t)
{
  o.t = nullptr;
}

template <typename T>
RetainPtr<T>::RetainPtr(const RetainPtr<T>& o)
  : RetainPtr(o.t)
{
}

template <typename T>
RetainPtr<T> adoptNS(T* t) {
#if __has_feature(objc_arc)
  return t;
#else
  return RetainPtr<T>(t, RetainPtr<T>::Adopt);
#endif
}

template <typename T>
RetainPtr<T> adoptCF(T t) {
  return RetainPtr<T>(t, RetainPtr<T>::Adopt);
}

template<typename T> inline RetainPtr<T> retainPtr(T ptr)
{
  return ptr;
}

template<typename T> inline RetainPtr<T> retainPtr(T* ptr)
{
  return ptr;
}

inline NSObject *bridge_cast(CFTypeRef object)
{
  return (__bridge NSObject *)object;
}

inline CFTypeRef bridge_cast(NSObject *object)
{
  return (__bridge CFTypeRef)object;
}

inline id bridge_id_cast(CFTypeRef object)
{
    return (__bridge id)object;
}

inline RetainPtr<id> bridge_id_cast(RetainPtr<CFTypeRef>&& object)
{
#if __has_feature(objc_arc)
    return adoptNS((__bridge_transfer id)object.leakRef());
#else
    return adoptNS((__bridge id)object.leakRef());
#endif
}

template <typename ExpectedType>
struct ObjCTypeCastTraits {
public:
    static bool isType(id object) { return [object isKindOfClass:[ExpectedType class]]; }

    template <typename ArgType>
    static bool isType(const ArgType *object) { return [object isKindOfClass:[ExpectedType class]]; }
};

template <typename ExpectedType, typename ArgType>
inline bool is_objc(ArgType * source)
{
    return source && ObjCTypeCastTraits<ExpectedType>::isType(source);
}

template<typename T> inline T *checked_objc_cast(id object)
{
    if (!object)
        return nullptr;

    if (!is_objc<T>(object))
      WTFCrash();

    return reinterpret_cast<T*>(object);
}

template<typename T, typename U> inline T *checked_objc_cast(U *object)
{
    if (!object)
        return nullptr;

    if (!is_objc<T>(object))
      WTFCrash();

    return static_cast<T*>(object);
}

template<typename T, typename U> RetainPtr<T> dynamic_objc_cast(RetainPtr<U>&& object)
{
    if (!is_objc<T>(object.get()))
        return nullptr;
    return adoptNS(static_cast<T*>(object.leakRef()));
}

template<typename T> RetainPtr<T> dynamic_objc_cast(RetainPtr<id>&& object)
{
    if (!is_objc<T>(object.get()))
        return nullptr;
    return adoptNS(reinterpret_cast<T*>(object.leakRef()));
}

template<typename T, typename U> RetainPtr<T> dynamic_objc_cast(const RetainPtr<U>& object)
{
    if (!is_objc<T>(object.get()))
        return nullptr;
    return static_cast<T*>(object.get());
}

template<typename T> RetainPtr<T> dynamic_objc_cast(const RetainPtr<id>& object)
{
    if (!is_objc<T>(object.get()))
        return nullptr;
    return reinterpret_cast<T*>(object.get());
}

template<typename T> T *dynamic_objc_cast(NSObject *object)
{
    if (!is_objc<T>(object))
        return nullptr;
    return static_cast<T*>(object);
}

template<typename T> T *dynamic_objc_cast(id object)
{
    if (!is_objc<T>(object))
        return nullptr;
    return reinterpret_cast<T*>(object);
}

template <typename> struct CFTypeTrait;

template<typename T> T dynamic_cf_cast(CFTypeRef object)
{
  if (!object)
    return nullptr;

  if (CFGetTypeID(object) != CFTypeTrait<T>::typeID())
    return nullptr;

  return static_cast<T>(const_cast<CF_BRIDGED_TYPE(id) void*>(object));
}

template<typename T> T checked_cf_cast(CFTypeRef object)
{
  if (!object)
    return nullptr;

  if (CFGetTypeID(object) != CFTypeTrait<T>::typeID())
    WTFCrash();

  return static_cast<T>(const_cast<CF_BRIDGED_TYPE(id) void*>(object));
}

template<typename T, typename U> RetainPtr<T> dynamic_cf_cast(RetainPtr<U>&& object)
{
  if (!object)
    return nullptr;

  if (CFGetTypeID(object.get()) != CFTypeTrait<T>::typeID())
    return nullptr;

  return adoptCF(static_cast<T>(const_cast<CF_BRIDGED_TYPE(id) void*>(object.leakRef())));
}

} // namespace WTF

#define WTF_DECLARE_CF_TYPE_TRAIT(ClassName) \
template <> \
struct WTF::CFTypeTrait<ClassName##Ref> { \
    static inline CFTypeID typeID(void) { return ClassName##GetTypeID(); } \
};

WTF_DECLARE_CF_TYPE_TRAIT(CFArray);
WTF_DECLARE_CF_TYPE_TRAIT(CFDictionary);

#define WTF_DECLARE_CF_MUTABLE_TYPE_TRAIT(ClassName, MutableClassName) \
template <> \
struct WTF::CFTypeTrait<MutableClassName##Ref> { \
    static inline CFTypeID typeID(void) { return ClassName##GetTypeID(); } \
};

WTF_DECLARE_CF_MUTABLE_TYPE_TRAIT(CFArray, CFMutableArray);
WTF_DECLARE_CF_MUTABLE_TYPE_TRAIT(CFDictionary, CFMutableDictionary);

using WTF::RetainPtr;
using WTF::adoptNS;
using WTF::adoptCF;
using WTF::retainPtr;
using WTF::downcast;
using WTF::bridge_cast;
using WTF::bridge_id_cast;
using WTF::is_objc;
using WTF::checked_objc_cast;
using WTF::dynamic_objc_cast;
using WTF::checked_cf_cast;
using WTF::dynamic_cf_cast;
