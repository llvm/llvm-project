// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s
// RUN: grep '.lazy_reference .objc_class_name_A' %t | count 1
// RUN: grep '.lazy_reference .objc_class_name_Unknown' %t | count 1
// RUN: grep '.lazy_reference .objc_class_name_Protocol' %t | count 1
// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -DWITH_IMPL -emit-llvm -o %t %s
// RUN: grep '.lazy_reference .objc_class_name_Root' %t | count 1

@interface Root
-(id) alloc;
-(id) init;
@end

@protocol P;

@interface A : Root
@end

@interface A (Category)
+(void) foo;
@end

#ifdef WITH_IMPL
@implementation A
@end
#endif

@interface Unknown
+test;
@end


int main() {
  id x = @protocol(P);
  [ A alloc ];
  [ A foo ];
  [ Unknown test ];
  return 0;
}

