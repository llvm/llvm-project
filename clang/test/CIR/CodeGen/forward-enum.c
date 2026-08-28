// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

extern enum X x;
void f(void) {
  x;
}

enum X {
  One,
  Two
};

// CIR: cir.global "private" external @x : !u32i
// CIR: cir.func{{.*}} @f
// CIR:   cir.get_global @x : !cir.ptr<!u32i>

// LLVM: @x = external global i32
// LLVM: define {{.*}}void @f()

// The same shape, but with a definition that contradicts the guess.  touch_v
// is what forces the conversion while the enum is incomplete, so uses after
// the definition have to be given the definition's type instead.
extern enum V v;
void touch_v(void) { v; }
enum V { VBig = 0x100000000 };
unsigned long read_v(void) { return v; }

// CIR: cir.func{{.*}} @touch_v()
// CIR: cir.func{{.*}} @read_v() -> !u64i
// CIR:   %[[VP:.+]] = cir.get_global @v : !cir.ptr<!u32i>
// CIR:   %[[VCAST:.+]] = cir.cast bitcast %[[VP]] : !cir.ptr<!u32i> -> !cir.ptr<!u64i>
// CIR:   %{{.+}} = cir.load align(8) %[[VCAST]] : !cir.ptr<!u64i>, !u64i

// LLVM: define {{.*}}void @touch_v()
// LLVM: define {{.*}}i64 @read_v()
// LLVM:   load i64, ptr @v, align 8

// A fixed underlying type is not a guess, so nothing here needs invalidating.
enum Y : long;
extern enum Y y;
void touch_y(void) { y; }
enum Y : long { YOne = 1 };
long read_y(void) { return y; }

// CIR: cir.func{{.*}} @touch_y()
// CIR:   cir.get_global @y : !cir.ptr<!s64i>
// CIR: cir.func{{.*}} @read_y() -> !s64i
// CIR:   cir.get_global @y : !cir.ptr<!s64i>

// LLVM: define {{.*}}void @touch_y()
// LLVM:   load i64, ptr @y, align 8
// LLVM: define {{.*}}i64 @read_y()
// LLVM:   load i64, ptr @y, align 8

// The function pointer forces the signature to be converted while the enum is
// incomplete, so the declaration keeps the guess.  The calls are emitted after
// the definition and use its type, bitcasting the callee to match.
enum W;
void takes_wider(enum W);
void (*wider_ptr)(enum W) = takes_wider;
enum W { WBig = 0x100000000 };
void use_wider(void) { takes_wider(WBig); }
void use_wider_ptr(void) { wider_ptr(WBig); }

// CIR: cir.func private @takes_wider(!u32i)
// CIR: cir.func{{.*}} @use_wider()
// CIR:   %[[WVAL:.+]] = cir.const #cir.int<4294967296> : !u64i
// CIR:   %[[WFN:.+]] = cir.get_global @takes_wider : !cir.ptr<!cir.func<(!u32i)>>
// CIR:   %[[WCAST:.+]] = cir.cast bitcast %[[WFN]] : !cir.ptr<!cir.func<(!u32i)>> -> !cir.ptr<!cir.func<(!u64i)>>
// CIR:   cir.call %[[WCAST]](%[[WVAL]]) : (!cir.ptr<!cir.func<(!u64i)>>, !u64i {llvm.noundef}) -> ()

// CIR: cir.func{{.*}} @use_wider_ptr()
// CIR:   %[[WPCAST:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!cir.ptr<!cir.func<(!u32i)>>> -> !cir.ptr<!cir.ptr<!cir.func<(!u64i)>>>
// CIR:   %[[WCALLEE:.+]] = cir.load align(8) %[[WPCAST]] : !cir.ptr<!cir.ptr<!cir.func<(!u64i)>>>, !cir.ptr<!cir.func<(!u64i)>>
// CIR:   cir.call %[[WCALLEE]](%{{.+}}) : (!cir.ptr<!cir.func<(!u64i)>>, !u64i {llvm.noundef}) -> ()

// LLVMCIR: declare void @takes_wider(i32)
// OGCG: declare void @takes_wider()
// LLVM: define {{.*}}void @use_wider()
// LLVM:   call void @takes_wider(i64 noundef 4294967296)
// LLVM: define {{.*}}void @use_wider_ptr()
// LLVM:   %[[WP:.+]] = load ptr, ptr @wider_ptr, align 8
// LLVM:   call void %[[WP]](i64 noundef 4294967296)

// Same, for a definition that keeps the guess's width but not its signedness.
enum S;
void takes_signed(enum S);
void (*signed_ptr)(enum S) = takes_signed;
enum S { SNeg = -1 };
void use_signed(void) { takes_signed(SNeg); }

// CIR: cir.func private @takes_signed(!u32i)
// CIR: cir.func{{.*}} @use_signed()
// CIR:   %[[SVAL:.+]] = cir.const #cir.int<-1> : !s32i
// CIR:   %[[SFN:.+]] = cir.get_global @takes_signed : !cir.ptr<!cir.func<(!u32i)>>
// CIR:   %[[SCAST:.+]] = cir.cast bitcast %[[SFN]] : !cir.ptr<!cir.func<(!u32i)>> -> !cir.ptr<!cir.func<(!s32i)>>
// CIR:   cir.call %[[SCAST]](%[[SVAL]]) : (!cir.ptr<!cir.func<(!s32i)>>, !s32i {llvm.noundef}) -> ()

// LLVMCIR: declare void @takes_signed(i32)
// OGCG: declare void @takes_signed()
// LLVM: define {{.*}}void @use_signed()
// LLVM:   call void @takes_signed(i32 noundef -1)

// A definition emitted after the enum completes takes its parameter type from
// the definition, so the call to it needs no bitcast.  defined_ptr is again
// what forces the early conversion.
enum D;
void takes_defined(enum D);
void (*defined_ptr)(enum D) = takes_defined;
enum D { DBig = 0x100000000 };
unsigned long sink;
void takes_defined(enum D d) { sink = d; }
void use_defined(void) { takes_defined(DBig); }

// CIR: cir.func{{.*}} @takes_defined(%arg0: !u64i {llvm.noundef}
// CIR: cir.func{{.*}} @use_defined()
// CIR:   %[[DVAL:.+]] = cir.const #cir.int<4294967296> : !u64i
// CIR:   cir.call @takes_defined(%[[DVAL]]) : (!u64i {llvm.noundef}) -> ()

// LLVM: define {{.*}}void @takes_defined(i64 noundef %{{.+}})
// LLVM: define {{.*}}void @use_defined()
// LLVM:   call void @takes_defined(i64 noundef 4294967296)
