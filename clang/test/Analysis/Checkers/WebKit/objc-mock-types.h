@class NSString;
@class NSArray;
@class NSMutableArray;
#define nil ((id)0)
#define CF_BRIDGED_TYPE(T) __attribute__((objc_bridge(T)))
#define CF_BRIDGED_MUTABLE_TYPE(T) __attribute__((objc_bridge_mutable(T)))
typedef CF_BRIDGED_TYPE(id) void * CFTypeRef;
typedef signed char BOOL;
typedef signed long CFIndex;
typedef unsigned long NSUInteger;
typedef const struct CF_BRIDGED_TYPE(id) __CFAllocator * CFAllocatorRef;
typedef const struct CF_BRIDGED_TYPE(NSString) __CFString * CFStringRef;
typedef const struct CF_BRIDGED_TYPE(NSArray) __CFArray * CFArrayRef;
typedef struct CF_BRIDGED_MUTABLE_TYPE(NSMutableArray) __CFArray * CFMutableArrayRef;
typedef struct CF_BRIDGED_MUTABLE_TYPE(CFRunLoopRef) __CFRunLoop * CFRunLoopRef;
extern const CFAllocatorRef kCFAllocatorDefault;
typedef struct _NSZone NSZone;
CFMutableArrayRef CFArrayCreateMutable(CFAllocatorRef allocator, CFIndex capacity);
extern void CFArrayAppendValue(CFMutableArrayRef theArray, const void *value);
CFArrayRef CFArrayCreate(CFAllocatorRef allocator, const void **values, CFIndex numValues);
CFIndex CFArrayGetCount(CFArrayRef theArray);
CFRunLoopRef CFRunLoopGetCurrent(void);
CFRunLoopRef CFRunLoopGetMain(void);
extern CFTypeRef CFRetain(CFTypeRef cf);
extern void CFRelease(CFTypeRef cf);
#define CFSTR(cStr) ((CFStringRef) __builtin___CFStringMakeConstantString ("" cStr ""))
extern Class NSClassFromString(NSString *aClassName);

__attribute__((objc_root_class))
@interface NSObject
+ (instancetype) alloc;
+ (Class) class;
+ (Class) superclass;
- (instancetype) init;
- (instancetype)retain;
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
+ (instancetype)dictionaryWithObjects:(const id [])objects forKeys:(const id <NSCopying> [])keys count:(NSUInteger)cnt;
@end

@interface NSArray : NSObject <NSCopying, NSFastEnumeration>
- (NSUInteger)count;
- (NSEnumerator *)objectEnumerator;
+ (NSArray *)arrayWithObjects:(const id [])objects count:(NSUInteger)count;
@end

@interface NSString : NSObject <NSCopying>
- (NSUInteger)length;
- (NSString *)stringByAppendingString:(NSString *)aString;
- ( const char *)UTF8String;
- (id)initWithUTF8String:(const char *)nullTerminatedCString;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end

@interface NSMutableString : NSString
@end

@interface NSValue : NSObject <NSCopying>
- (void)getValue:(void *)value;
@end

@interface NSNumber : NSValue
- (char)charValue;
- (id)initWithInt:(int)value;
+ (NSNumber *)numberWithInt:(int)value;
@end

@interface SomeObj : NSObject
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
  RetainPtr &operator=(PtrType t) {
    RetainPtr o(t);
    swap(o);
    return *this;
  }
  PtrType leakRef()
  {
    PtrType s = t;
    t = nullptr;
    return s;
  }
  operator PtrType() const { return t; }
  operator bool() const { return t; }

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

}

using WTF::RetainPtr;
using WTF::adoptNS;
using WTF::adoptCF;
using WTF::retainPtr;
using WTF::downcast;
using WTF::bridge_cast;