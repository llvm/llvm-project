@interface MyClass
+(void)meth;
@end

#define MACRO2(x) (x)
#define MACRO(x) MACRO2((x))

void test() {
  MACRO([MyClass meth]);
}

#define INVOKE(METHOD, CLASS) [CLASS METHOD]

void test2() {
  INVOKE(meth, MyClass);
}
