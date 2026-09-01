// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios14.0 -fobjc-runtime=ios-14.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -emit-llvm -o - %s | FileCheck %s

// The constant dictionary emitter sorts string keys by UTF-16 code unit, which
// is the order the runtime uses to look them up. This matters for keys that mix
// the BMP above the surrogate range (U+E000..U+FFFF) with astral characters
// (U+10000..U+10FFFF): UTF-16 encodes astral characters with lead surrogates
// (0xD800..0xDBFF) that sort *below* 0xE000, which is the opposite of their
// UTF-8 byte order. Sorting by UTF-16 here keeps compile-time emission and
// runtime lookup consistent so the keys can be found.

#if __LP64__
typedef unsigned long NSUInteger;
#else
typedef unsigned int NSUInteger;
#endif

@interface NSNumber
+ (NSNumber *)numberWithInt:(int)value;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id[])objects forKeys:(const id[])keys count:(NSUInteger)cnt;
@end

// The emoji U+1F600 is stored as the UTF-16 surrogate pair <0xD83D, 0xDE00>.
// CHECK: @.str = private unnamed_addr constant [3 x i16] [i16 -10179, i16 -8704, i16 0], section "__TEXT,__ustring"
// CHECK: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 {{[0-9]+}}, ptr @.str, i64 2 }

// The Private Use Area character U+E000 is a single UTF-16 code unit <0xE000>.
// CHECK: @.str.2 = private unnamed_addr constant [2 x i16] [i16 -8192, i16 0], section "__TEXT,__ustring"
// CHECK: @_unnamed_cfstring_.3 = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 {{[0-9]+}}, ptr @.str.2, i64 1 }

// The emitted keys array is ordered by UTF-16 code unit: the emoji's lead
// surrogate 0xD83D sorts before the PUA's 0xE000, so @_unnamed_cfstring_ (emoji)
// comes first even though its UTF-8 bytes (F0 9F 98 80) are greater than the
// PUA's (EE 80 80). This matches the runtime's UTF-16 lookup order.
// CHECK: @_unnamed_array_storage = internal unnamed_addr constant [2 x ptr] [ptr @_unnamed_cfstring_, ptr @_unnamed_cfstring_.3]
// CHECK: @_unnamed_nsdictionary_ = private constant %struct.__builtin_NSDictionary { ptr @"OBJC_CLASS_$_NSConstantDictionary", i64 1, i64 2, ptr @_unnamed_array_storage, ptr @_unnamed_array_storage.5 }

static NSDictionary *const diverges = @{
    @"\U0001F600" : @1,
    @"\uE000" : @2,
};

const void *use(void) { return (const void *)diverges; }
