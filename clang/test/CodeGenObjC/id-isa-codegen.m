// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5  -emit-llvm -o - %s | FileCheck -check-prefix CHECK-LP64 %s
// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5  -emit-llvm -o - %s | FileCheck -check-prefix CHECK-LP32 %s

typedef struct objc_class *Class;

typedef struct objc_object {
    Class isa;
} *id;

@interface I
+ (Class) class;
- (void)meth : (id)object : (id)src_object;
+ (unsigned char) isSubclassOfClass:(Class)aClass ;
@end

@implementation I
+ (Class) class {return 0;}
+ (unsigned char) isSubclassOfClass:(Class)aClass {return 0;}
- (void)meth : (id)object  : (id)src_object {
    [object->isa isSubclassOfClass:[I class]];

    [(*object).isa isSubclassOfClass:[I class]];

    object->isa = src_object->isa;
    (*src_object).isa = (*object).isa;
}
@end


static Class MyClass;

Class Test(const void *inObject1) {
  if(((id)inObject1)->isa == MyClass)
   return ((id)inObject1)->isa;
  return (id)0;
}

@interface Foo { 
@public 
  id isa; 
} 
+(id)method;
@end

id Test2(void) {
    if([Foo method]->isa)
      return (*[Foo method]).isa;
    return [Foo method]->isa;
}

@interface Cat   {}
@end

@interface SuperCat : Cat {}
+(void)geneticallyAlterCat:(Cat *)cat;
@end

@implementation SuperCat
+ (void)geneticallyAlterCat:(Cat *)cat {
    Class dynamicSubclass;
    ((id)cat)->isa = dynamicSubclass;
}
@end
// CHECK-LP64: %{{.*}} = load ptr, ptr %
// CHECK-NEXT: store ptr %{{.*}}, ptr %{{.*}}

// CHECK-LP32: %{{.*}} = load ptr, ptr %
// CHECK-NEXT: store ptr %{{.*}}, ptr %{{.*}}
