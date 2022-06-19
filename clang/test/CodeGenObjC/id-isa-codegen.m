// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5  -emit-llvm -o - %s | FileCheck -check-prefix CHECK-LP64 %s
// RUN: %clang_cc1 -no-opaque-pointers -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5  -emit-llvm -o - %s | FileCheck -check-prefix CHECK-LP32 %s

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


// rdar 7470820
static Class MyClass;

Class Test(const void *inObject1) {
  if(((id)inObject1)->isa == MyClass)
   return ((id)inObject1)->isa;
  return (id)0;
}

// rdar 7609722
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

// rdar 7709015
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
// CHECK-LP64: %{{.*}} = load i8*, i8** %
// CHECK-NEXT: %{{.*}} = bitcast i8* %{{.*}} to i8**
// CHECK-NEXT: store i8* %{{.*}}, i8** %{{.*}}

// CHECK-LP32: %{{.*}} = load i8*, i8** %
// CHECK-NEXT: %{{.*}} = bitcast i8* %{{.*}} to i8**
// CHECK-NEXT: store i8* %{{.*}}, i8** %{{.*}}
