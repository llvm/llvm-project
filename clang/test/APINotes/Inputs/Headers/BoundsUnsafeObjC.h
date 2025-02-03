@interface Foo
- (void) asdf_counted: (int *)buf len: (int)len;
- (void) asdf_sized: (int *)buf size: (int)size;
- (void) asdf_counted_n: (int *)buf len: (int)len;
- (void) asdf_sized_n: (int *)buf size: (int)size;
- (void) asdf_ended: (int *)buf end: (int *)end;

- (void) asdf_sized_mul: (int *)buf size:(int)size count:(int)count;
- (void) asdf_counted_out: (int **)buf len:(int *)len;
- (void) asdf_counted_const: (int *)buf;
- (void) asdf_counted_nullable: (int)len buf:(int * _Nullable)buf;
- (void) asdf_counted_noescape: (int *)buf len: (int)len;

- (void) asdf_nterm: (char *) buf;
@end

