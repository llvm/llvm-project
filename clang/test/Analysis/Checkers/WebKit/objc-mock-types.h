@class NSString;
@class NSArray;
@class NSMutableArray;
#define CF_BRIDGED_TYPE(T) __attribute__((objc_bridge(T)))
typedef CF_BRIDGED_TYPE(id) void * CFTypeRef;
typedef signed long CFIndex;
typedef const struct __CFAllocator * CFAllocatorRef;
typedef const struct CF_BRIDGED_TYPE(NSString) __CFString * CFStringRef;
typedef const struct CF_BRIDGED_TYPE(NSArray) __CFArray * CFArrayRef;
typedef struct CF_BRIDGED_TYPE(NSMutableArray) __CFArray * CFMutableArrayRef;
extern const CFAllocatorRef kCFAllocatorDefault;
CFMutableArrayRef CFArrayCreateMutable(CFAllocatorRef allocator, CFIndex capacity);
extern void CFArrayAppendValue(CFMutableArrayRef theArray, const void *value);
CFArrayRef CFArrayCreate(CFAllocatorRef allocator, const void **values, CFIndex numValues);
CFIndex CFArrayGetCount(CFArrayRef theArray);
extern CFTypeRef CFRetain(CFTypeRef cf);
extern void CFRelease(CFTypeRef cf);

__attribute__((objc_root_class))
@interface NSObject
+ (instancetype) alloc;
- (instancetype) init;
- (instancetype)retain;
- (void)release;
@end

@interface SomeObj : NSObject
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
      CFRetain(t);
  }
  RetainPtr(RetainPtr&& o)
    : RetainPtr(o.t)
  {
    o.t = nullptr;
  }
  RetainPtr(const RetainPtr& o)
    : RetainPtr(o.t)
  {
  }
  RetainPtr operator=(const RetainPtr& o)
  {
    if (t)
      CFRelease(t);
    t = o.t;
    if (t)
      CFRetain(t);
    return *this;
  }
  ~RetainPtr() {
    clear();
  }
  void clear() {
    if (t)
      CFRelease(t);
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
  operator PtrType() const { return t; }
  operator bool() const { return t; }

private:
  template <typename U> friend RetainPtr<U> adoptNS(U*);
  template <typename U> friend RetainPtr<U> adoptCF(U);

  enum AdoptTag { Adopt };
  RetainPtr(PtrType t, AdoptTag) : t(t) { }
};

template <typename T>
RetainPtr<T> adoptNS(T* t) {
  return RetainPtr<T>(t, RetainPtr<T>::Adopt);
}

template <typename T>
RetainPtr<T> adoptCF(T t) {
  return RetainPtr<T>(t, RetainPtr<T>::Adopt);
}

}

using WTF::RetainPtr;
using WTF::adoptNS;
using WTF::adoptCF;
using WTF::downcast;
