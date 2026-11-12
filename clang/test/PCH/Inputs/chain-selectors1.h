@interface X
  -(void)f;
  -(void)f2;
  -(void)g:(int)p;
  -(void)h:(int)p1 foo:(int)p2;
@end

void foo1(void) {
  // FIXME: Can't verify warnings in headers
  //(void)@selector(x);
  (void)@selector(f);
}

@interface X (Blah)
- (void)blah_method;
@end
