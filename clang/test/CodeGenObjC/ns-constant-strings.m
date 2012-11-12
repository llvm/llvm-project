// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fno-constant-cfstrings -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix CHECK-FRAGILE < %t %s

// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fno-constant-cfstrings -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix CHECK-NONFRAGILE < %t %s

@interface NSString @end

@interface NSSimpleCString : NSString {
@protected
    char *bytes;
    unsigned int numBytes;
}
@end
    
@interface NSConstantString : NSSimpleCString
@end

#if OBJC_API_VERSION >= 2
extern Class _NSConstantStringClassReference;
#else
extern struct objc_class _NSConstantStringClassReference;
#endif

const NSConstantString *appKey =  @"MyApp";

int main() {
  const NSConstantString *appKey =  @"MyApp";
  const NSConstantString *appKey1 =  @"MyApp1";
}

// CHECK-FRAGILE: @_NSConstantStringClassReference = external global
// CHECK-NONFRAGILE: @"OBJC_CLASS_$_NSConstantString" = external global

// CHECK-FRAGILE: @.str = private unnamed_addr constant [6 x i8] c"MyApp\00"
// CHECK-FRAGILE: @.str1 = private unnamed_addr constant [7 x i8] c"MyApp1\00"

// CHECK-NONFRAGILE: @.str = private unnamed_addr constant [6 x i8] c"MyApp\00"
// CHECK-NONFRAGILE: @.str1 = private unnamed_addr constant [7 x i8] c"MyApp1\00"
