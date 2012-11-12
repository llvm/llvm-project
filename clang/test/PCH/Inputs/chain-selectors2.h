@interface Y
  -(void)f;
  -(double)f2;
  -(void)e;
@end

void foo2() {
  // FIXME: Can't verify warnings in headers
  //(void)@selector(y);
  //(void)@selector(e);
}

@interface X (Blarg)
- (void)blarg_method;
@end
