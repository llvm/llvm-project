@interface B
@end


@interface A {
@public
  union {
    struct {
      B *b;
    } a_b;
    void *void_ptr;
  } data;
}
@end
