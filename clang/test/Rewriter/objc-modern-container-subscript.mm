// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar://11203853

typedef unsigned long size_t;

void *sel_registerName(const char *);

@protocol P @end

@interface NSMutableArray
#if __has_feature(objc_subscripting)
- (id)objectAtIndexedSubscript:(size_t)index;
- (void)setObject:(id)object atIndexedSubscript:(size_t)index;
#endif
@end

#if __has_feature(objc_subscripting)
@interface XNSMutableArray
- (id)objectAtIndexedSubscript:(size_t)index;
- (void)setObject:(id)object atIndexedSubscript:(size_t)index;
#endif
@end

@interface NSMutableDictionary
- (id)objectForKeyedSubscript:(id)key;
- (void)setObject:(id)object forKeyedSubscript:(id)key;
@end

@class NSString;

int main() {
  NSMutableArray<P> * array;
  id oldObject = array[10];

  array[10] = oldObject;

  id unknown_array;
  oldObject = unknown_array[1];

  unknown_array[1] = oldObject;

  NSMutableDictionary *dictionary;
  NSString *key;
  id newObject;
  oldObject = dictionary[key];
  dictionary[key] = newObject;  // replace oldObject with newObject
}

