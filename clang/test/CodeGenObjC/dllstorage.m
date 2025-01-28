// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fdeclspec -fobjc-runtime=ios -fobjc-exceptions -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap -check-prefix CHECK-IR %s
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fdeclspec -fobjc-runtime=gnustep-2.0 -fobjc-exceptions -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap -check-prefix CHECK-NF %s
// RUN: %clang_cc1 -triple i686-windows-itanium -fms-extensions -fobjc-runtime=macosx -fdeclspec -fobjc-exceptions -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap -check-prefix CHECK-IR %s
// RUN: %clang_cc1 -triple i686-windows-itanium -fms-extensions -fobjc-runtime=objfw -fdeclspec -fobjc-exceptions -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap -check-prefix CHECK-FW %s

// CHECK-IR-DAG: @_objc_empty_cache = external dllimport global %struct._objc_cache

__declspec(dllimport)
@interface I
+ (instancetype) new;
@end

// CHECK-IR-DAG: @"OBJC_METACLASS_$_I" = external dllimport global %struct._class_t
// CHECK-IR-DAG: @"OBJC_CLASS_$_I" = external dllimport global %struct._class_t

// CHECK-NF-DAG: @"$_OBJC_CLASS_I" = external dllimport global ptr

__declspec(dllexport)
@interface J : I
@end

// CHECK-IR-DAG: @"OBJC_METACLASS_$_J" = dso_local dllexport global %struct._class_t
// CHECK-IR-DAG: @"OBJC_CLASS_$_J" = dso_local dllexport global %struct._class_t

// CHECK-FW-DAG: @_OBJC_METACLASS_J = dso_local dllexport global
// CHECK-FW-DAG: @_OBJC_CLASS_J = dso_local dllexport global

// CHECK-NF-DAG: @"$_OBJC_METACLASS_J" = internal global
// CHECK-NF-DAG: @"$_OBJC_CLASS_J" = dllexport global

@implementation J {
  id _ivar;
}
@end

// CHECK-IR-DAG: @"OBJC_IVAR_$_J._ivar" = global i32

// CHECK-NF-DAG: @"__objc_ivar_offset_J._ivar.@" = hidden global i32

@interface K : J
@end

// CHECK-IR-DAG: @"OBJC_METACLASS_$_K" = dso_local global %struct._class_t
// CHECK-IR-DAG: @"OBJC_CLASS_$_K" = dso_local global %struct._class_t

// CHECK-FW-DAG: @_OBJC_METACLASS_K = dso_local global
// CHECK-FW-DAG: @_OBJC_CLASS_K = dso_local global

// CHECK-NF-DAG: @"$_OBJC_METACLASS_K" = internal global
// CHECK-NF-DAG: @"$_OBJC_CLASS_K" = global

@implementation K {
  id _ivar;
}
@end

// CHECK-IR-DAG: @"OBJC_IVAR_$_K._ivar" = global i32

// CHECK-NF-DAG: @"__objc_ivar_offset_K._ivar.@" = hidden global i32

__declspec(dllexport)
@interface L : K
@end

// CHECK-IR-DAG: @"OBJC_METACLASS_$_L" = dso_local dllexport global %struct._class_t
// CHECK-IR-DAG: @"OBJC_CLASS_$_L" = dso_local dllexport global %struct._class_t

// CHECK-FW-DAG: @_OBJC_METACLASS_L = dso_local dllexport global
// CHECK-FW-DAG: @_OBJC_CLASS_L = dso_local dllexport global

// CHECK-NF-DAG: @"$_OBJC_METACLASS_L" = internal global
// CHECK-NF-DAG: @"$_OBJC_CLASS_L" = dllexport global

@implementation L {
  id _none;

  @public
  id _public;

  @protected
  id _protected;

  @package
  id _package;

  @private
  id _private;
}
@end

// CHECK-IR-DAG: @"OBJC_IVAR_$_L._none" = global i32
// CHECK-IR-DAG: @"OBJC_IVAR_$_L._public" = dllexport global i32
// CHECK-IR-DAG: @"OBJC_IVAR_$_L._protected" = dllexport global i32
// CHECK-IR-DAG: @"OBJC_IVAR_$_L._package" = global i32
// CHECK-IR-DAG: @"OBJC_IVAR_$_L._private" = global i32

// CHECK-NF-DAG: @"__objc_ivar_offset_L._none.@" = hidden global i32
// CHECK-NF-DAG: @"__objc_ivar_offset_L._public.@" = dso_local dllexport global i32
// CHECK-NF-DAG: @"__objc_ivar_offset_L._protected.@" = dso_local dllexport global i32
// CHECK-NF-DAG: @"__objc_ivar_offset_L._package.@" = hidden global i32
// CHECK-NF-DAG: @"__objc_ivar_offset_L._private.@" = hidden global i32

__declspec(dllimport)
@interface M : I {
  @public
  id _ivar;
}
@end

// CHECK-FW-DAG: @_OBJC_CLASS_M = external dllimport global i32

// CHECK-IR-DAG: @"OBJC_IVAR_$_M._ivar" = external dllimport global i32

// CHECK-NF-DAG: @"$_OBJC_REF_CLASS_M" = external dllimport global ptr
// CHECK-NF-DAG: @"__objc_ivar_offset_M._ivar.@" = external dllimport global i32

__declspec(dllexport)
__attribute__((__objc_exception__))
@interface N : I
@end

// CHECK-FW-DAG: @_OBJC_METACLASS_N = dso_local dllexport global
// CHECK-FW-DAG: @_OBJC_CLASS_N = dso_local dllexport global

// CHECK-NF-DAG: @"$_OBJC_CLASS_N" = dllexport global

@implementation N : I
@end

// CHECK-IR-DAG: @"OBJC_EHTYPE_$_N" = dso_local dllexport global %struct._objc_typeinfo

__declspec(dllimport)
__attribute__((__objc_exception__))
@interface O : I
@end

// CHECK-IR-DAG: @"OBJC_EHTYPE_$_O" = external dllimport global %struct._objc_typeinfo

__attribute__((__objc_exception__))
@interface P : I
@end

// CHECK-IR-DAG: @"OBJC_EHTYPE_$_P" = external dso_local global %struct._objc_typeinfo

@interface Q : M
@end

id f(Q *q) {
  return q->_ivar;
}

// CHECK-IR-DAG: @"OBJC_IVAR_$_M._ivar" = external dllimport global i32

// CHECK-NF-DAG: @"__objc_ivar_offset_M._ivar.@" = external dllimport global i32

int g(void) {
  @autoreleasepool {
    M *mi = [M new];
    @try {
      mi->_ivar = (void *)0;
      @throw(@"CFConstantString");
    } @catch (id) {
      return 1;
    } @catch (I *) {
      return 2;
    } @catch (J *) {
      return 3;
    } @catch (K *) {
      return 4;
    } @catch (L *) {
      return 5;
    } @catch (M *) {
      return 6;
    } @catch (N *) {
      return 7;
    } @catch (O *) {
      return 8;
    } @catch (P *) {
      return 9;
    }
  }
  return 0;
}

// CHECK-IR-DAG: @OBJC_EHTYPE_id = external dllimport global %struct._objc_typeinfo
// CHECK-IR-DAG: @"OBJC_EHTYPE_$_I" = weak global %struct._objc_typeinfo
// CHECK-IR-DAG: @"OBJC_EHTYPE_$_K" = weak global %struct._objc_typeinfo
// CHECK-IR-DAG: @"OBJC_EHTYPE_$_L" = weak global %struct._objc_typeinfo
// CHECK-IR-DAG: @"OBJC_EHTYPE_$_M" = weak global %struct._objc_typeinfo

