// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -fno-constant-cfstrings -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios14.0 -fobjc-runtime=ios-14.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -fno-constant-cfstrings -emit-llvm -o - %s | FileCheck %s

// With -fno-constant-cfstrings the keys are emitted as
// OBJC_CLASS_$_NSConstantString (raw UTF-8 bytes preserved) and compared as
// raw bytes at lookup time, so the emitter must sort by UTF-8 byte order
// (LKS < RKS) rather than UTF-16 code-unit order. This is the opposite of the
// default order for keys mixing the BMP above the surrogate range with astral
// characters: the PUA key (bytes EE 80 80) sorts before the emoji (bytes
// F0 9F 98 80), even though the emoji's UTF-16 lead surrogate (0xD83D) sorts
// before the PUA's code unit (0xE000).

#if __LP64__
typedef unsigned long NSUInteger;
#else
typedef unsigned int NSUInteger;
#endif

@interface NSString @end

@interface NSSimpleCString : NSString {
@protected
    char *bytes;
    unsigned int numBytes;
}
@end

@interface NSConstantString : NSSimpleCString
@end

@interface NSNumber
+ (NSNumber *)numberWithInt:(int)value;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id[])objects forKeys:(const id[])keys count:(NSUInteger)cnt;
@end

// The emoji U+1F600 is stored as the raw UTF-8 bytes F0 9F 98 80.
// CHECK: @.str = private unnamed_addr constant [5 x i8] c"\F0\9F\98\80\00"
// CHECK: @_unnamed_nsstring_ = private constant %struct.__builtin_NSString { ptr @"OBJC_CLASS_$_NSConstantString", ptr @.str, i32 4 }

// The Private Use Area character U+E000 is stored as the raw UTF-8 bytes EE 80 80.
// CHECK: @.str.{{[0-9]+}} = private unnamed_addr constant [4 x i8] c"\EE\80\80\00"
// CHECK: @_unnamed_nsstring_.{{[0-9]+}} = private constant %struct.__builtin_NSString { ptr @"OBJC_CLASS_$_NSConstantString", ptr @.str.{{[0-9]+}}, i32 3 }

// The emitted keys array is ordered by UTF-8 byte order: the PUA's first byte
// 0xEE sorts before the emoji's first byte 0xF0, so the PUA key (suffixed
// global, emitted second) comes first in the array.
// CHECK: @_unnamed_array_storage = internal unnamed_addr constant [2 x ptr] [ptr @_unnamed_nsstring_.{{[0-9]+}}, ptr @_unnamed_nsstring_]
static NSDictionary *const diverges = @{
    @"\U0001F600" : @1,
    @"\uE000" : @2,
};

const void *use(void) { return (const void *)diverges; }
