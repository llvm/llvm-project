@interface Y
  -(void)f;
  -(void)f2;
  -(void)x;
  -(void)y;
  -(void)e;
@end

void foo2(void) {
  // FIXME: Can't verify warnings in headers
  //(void)@selector(y);
  //(void)@selector(e);
}

@interface X (Blarg)
- (void)blarg_method;
@end
