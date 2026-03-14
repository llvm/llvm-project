static inline void *test0(id x) {
  return x;
}

static inline void **test1(__strong id* x) {
  return (void**) x;
}





struct Test3 {
  id *field;
};

@interface Test4 {
@public
  id *field1;
  __strong id *field2;
}
@end

struct Test5 {
  id field;
};







extern struct Test6 *const kMagicConstant;





@interface Test7
@property id *prop;
@end







static inline void *test8(id ptr) {
  return (__bridge_retain void*) ptr;
}

typedef struct {
  const char *name;
  id field;
} Test9;
