@protocol P @end

@interface NSMutableArray
- (id)objectAtIndexedSubscript:(unsigned int)index;
- (void)setObject:(id)object atIndexedSubscript:(unsigned int)index;
@end

@interface NSMutableDictionary
- (id)objectForKeyedSubscript:(id)key;
- (void)setObject:(id)object forKeyedSubscript:(id)key;
@end

void all(void) {
  NSMutableArray *array;
  id oldObject = array[10];

  array[10] = oldObject;

  NSMutableDictionary *dictionary;
  id key;
  id newObject;
  oldObject = dictionary[key];

  dictionary[key] = newObject;
}

