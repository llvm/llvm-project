// RUN: %clang_cc1 %s -fsyntax-only

@interface NSObject
+ alloc;
- init;
@end

struct D {
  double d;
};

@interface Foo : NSObject 

- method:(int)a;
- method:(int)a;

@end

@interface Bar : NSObject 

- method:(void *)a;

@end

@interface Car : NSObject 

- method:(struct D)a;

@end

@interface Zar : NSObject 

- method:(float)a;

@end

@interface Rar : NSObject 

- method:(float)a;

@end

int main() {
  id xx = [[Car alloc] init]; // expected-warning {{incompatible types assigning 'int' to 'id'}}

  [xx method:4];
}
