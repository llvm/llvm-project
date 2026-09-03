// REQUIRES: x86-registered-target, system-darwin, lld
//
// End-to-end test:
// - clang produces ObjC bitcode object files,
// - ld64.lld links and runs ObjCConstantIvarOffsetPass under LTO,
// - llvm-objdump confirms both the presliding and constant ivar offset folding
//
//   NSObject(size 8)
//     `- Super(start 8, size 16: _x at 8, _hidden at 12)
//          `- Sub(local start 12, size 16 -> resolved start 16, size 20)
//
// RUN: rm -rf %t && split-file %s %t
//
// --- ThinLTO path ---
// RUN: %clang -target x86_64-apple-macosx10.15.0 \
// RUN:     -flto=thin -O1 -c %t/super.m -o %t/super.thin.o
// RUN: %clang -target x86_64-apple-macosx10.15.0 \
// RUN:     -flto=thin -O1 -c %t/sub.m -o %t/sub.thin.o
// RUN: %clang -target x86_64-apple-macosx10.15.0 \
// RUN:     -flto=thin -O1 -fuse-ld=lld -dynamiclib -Wl,-undefined,dynamic_lookup \
// RUN:     %t/super.thin.o %t/sub.thin.o -o %t/thin.dylib
// RUN: llvm-objdump -d --x86-asm-syntax=intel %t/thin.dylib | FileCheck %s --check-prefix=ASM
// RUN: llvm-objdump --syms --full-contents %t/thin.dylib | FileCheck %s --check-prefix=DATA
//
// --- FullLTO path ---
// RUN: %clang -target x86_64-apple-macosx10.15.0 \
// RUN:     -flto -O1 -c %t/super.m -o %t/super.full.o
// RUN: %clang -target x86_64-apple-macosx10.15.0 \
// RUN:     -flto -O1 -c %t/sub.m -o %t/sub.full.o
// RUN: %clang -target x86_64-apple-macosx10.15.0 \
// RUN:     -flto -O1 -fuse-ld=lld -dynamiclib -Wl,-undefined,dynamic_lookup \
// RUN:     %t/super.full.o %t/sub.full.o -o %t/full.dylib
// RUN: llvm-objdump -d --x86-asm-syntax=intel %t/full.dylib | FileCheck %s --check-prefix=ASM
// RUN: llvm-objdump --syms --full-contents %t/full.dylib | FileCheck %s --check-prefix=DATA

// Check that ivar accesses use folded immediate offsets.
// Before: mov rax, qword ptr [rip + 0x...] ## 0x... <_OBJC_IVAR_$_Sub._y>
//         mov eax, dword ptr [r14 + rax]
// After:  mov eax, dword ptr [r14 + 0x10]
//
// Ivars are always [self + offset]; matching the addressing mode suffices.
// ASM-LABEL: <-[Super bump:]>:
// ASM-NOT:   ptr{{.*}}## {{.*}}<_OBJC_IVAR_$_{{.*}}>
// ASM:       dword ptr [r{{[a-z0-9]+}} + 0x8]
// ASM:       dword ptr [r{{[a-z0-9]+}} + 0xc]
//
// ASM-LABEL: <-[Sub combined:]>:
// ASM-NOT:   ptr{{.*}}## {{.*}}<_OBJC_IVAR_$_{{.*}}>
// ASM:       dword ptr [r{{[a-z0-9]+}} + 0x10]

// Check that ivar offset globals hold correctly preslid values.
// DATA:      SYMBOL TABLE:
// DATA-DAG:  [[#%x, SUPER_X:]] {{.*}} __DATA,__objc_ivar _OBJC_IVAR_$_Super._x
// DATA-DAG:  [[#%x, SUPER_H:]] {{.*}} __DATA,__objc_ivar _OBJC_IVAR_$_Super._hidden
// DATA-DAG:  [[#%x, SUB_Y:]]   {{.*}} __DATA,__objc_ivar _OBJC_IVAR_$_Sub._y
//
// DATA:      Contents of section __DATA,__objc_ivar:
// _x=8, _hidden=12 on one row; _y=16 (preslid from 8).
// DATA-NEXT: {{0*}}[[#SUPER_X]] 08000000 00000000 0c000000 00000000
// DATA-NEXT: {{0*}}[[#SUB_Y]]   10000000 00000000

//--- super.m
#import <objc/NSObject.h>

@interface Super : NSObject
@property(nonatomic) int x;
- (int)bump:(int)delta;
@end

@interface Super ()
@property(nonatomic) int hidden;
@end

@implementation Super
- (int)bump:(int)delta {
  _x += delta;
  _hidden += delta > 0 ? delta : -delta;
  return _x + _hidden;
}
@end

//--- sub.m
#import <objc/NSObject.h>

@interface Super : NSObject
@property(nonatomic) int x;
@end

@interface Sub : Super
@property(nonatomic) int y;
- (int)combined:(int)count;
@end

@implementation Sub
- (int)combined:(int)count {
  int total = self.x + _y;
  for (int i = 0; i < count; ++i)
    total += _y + i;
  return total;
}
@end
