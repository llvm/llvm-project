// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin %s -o /dev/null

typedef unsigned int size_t;
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

int main(void) {
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
  dictionary[key] = newObject;	// replace oldObject with newObject

}

