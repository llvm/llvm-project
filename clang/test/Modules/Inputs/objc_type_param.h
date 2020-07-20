__attribute__((objc_root_class))
@interface Root {
  Class isa;
}
@end

@interface A<T,U> : Root
@end

@interface B<T,U> : A<T,U>
typedef void (*BCallback)(T, U);
+ (id) newWithCallback: (BCallback) callback;
@end
