// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -no-enable-noundef-analysis -fclangir -emit-llvm %s -o %t-cir-nonoundef.ll
// RUN: FileCheck --check-prefix=NONOUNDEF --input-file=%t-cir-nonoundef.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -no-enable-noundef-analysis -emit-llvm %s -o %t-nonoundef.ll
// RUN: FileCheck --check-prefix=NONOUNDEF --input-file=%t-nonoundef.ll %s

// A transparent union is passed as its first member.  It is noundef only where
// that member is extended.

typedef union {
  short s;
  unsigned short us;
} __attribute__((transparent_union)) ShortFirst;

typedef union {
  unsigned char c;
  signed char sc;
} __attribute__((transparent_union)) UCharFirst;

typedef union {
  int i;
  unsigned u;
} __attribute__((transparent_union)) IntFirst;

typedef union {
  int *ip;
  const char *cp;
} __attribute__((transparent_union)) PtrFirst;

// CIR: cir.record_layouts = {
// CIR-SAME: IntFirst = #cir.record_layout<{{[^>]*}}, transparent_union = true>
// CIR-SAME: PtrFirst = #cir.record_layout<{{[^>]*}}, transparent_union = true>
// CIR-SAME: ShortFirst = #cir.record_layout<{{[^>]*}}, transparent_union = true>
// CIR-SAME: UCharFirst = #cir.record_layout<{{[^>]*}}, transparent_union = true>

void take_short(ShortFirst u) { (void)u; }

// CIR:      cir.func {{.*}} @take_short(%[[ARG:.*]]: !s16i {llvm.noundef, llvm.signext}
// CIR:        %[[COERCE:.*]] = cir.alloca "coerce" align(2) : !cir.ptr<!s16i>
// CIR:        cir.store %[[ARG]], %[[COERCE]] : !s16i, !cir.ptr<!s16i>
// CIR:        cir.cast bitcast %[[COERCE]] : !cir.ptr<!s16i> -> !cir.ptr<!rec_ShortFirst>

// LLVM: define dso_local void @take_short(i16 noundef signext %{{.*}})
// NONOUNDEF: define dso_local void @take_short(i16 signext %{{.*}})
void take_uchar(UCharFirst u) { (void)u; }

// CIR: cir.func {{.*}} @take_uchar(%{{.*}}: !u8i {llvm.noundef, llvm.zeroext}

// LLVM: define dso_local void @take_uchar(i8 noundef zeroext %{{.*}})
// NONOUNDEF: define dso_local void @take_uchar(i8 zeroext %{{.*}})

void take_int(IntFirst u) { (void)u; }

// CIR: cir.func {{.*}} @take_int(%{{.*}}: !s32i loc

// LLVM: define dso_local void @take_int(i32 %{{.*}})

void take_ptr(PtrFirst u) { (void)u; }

// CIR: cir.func {{.*}} @take_ptr(%{{.*}}: !cir.ptr<!void> loc

// LLVM: define dso_local void @take_ptr(ptr %{{.*}})

void ext_short(ShortFirst u);
void ext_ptr(PtrFirst u);

void callers(short s, int *p) {
  ext_short(s);
  ext_ptr(p);
}

// CIR:      cir.func {{.*}} @callers(
// CIR:        %[[SHORT_TMP:.*]] = cir.alloca "agg.tmp0" align(2) : !cir.ptr<!rec_ShortFirst>
// CIR:        %[[PTR_TMP:.*]] = cir.alloca "agg.tmp1" align(8) : !cir.ptr<!rec_PtrFirst>
// CIR:        %[[SHORT_ADDR:.*]] = cir.cast bitcast %[[SHORT_TMP]] : !cir.ptr<!rec_ShortFirst> -> !cir.ptr<!s16i>
// CIR:        %[[SHORT:.*]] = cir.load align(2) %[[SHORT_ADDR]] : !cir.ptr<!s16i>, !s16i
// CIR:        cir.call @ext_short(%[[SHORT]]) : (!s16i {llvm.noundef, llvm.signext}) -> ()
// CIR:        %[[PTR_ADDR:.*]] = cir.cast bitcast %[[PTR_TMP]] : !cir.ptr<!rec_PtrFirst> -> !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[PTR:.*]] = cir.load align(8) %[[PTR_ADDR]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:        cir.call @ext_ptr(%[[PTR]]) : (!cir.ptr<!void>) -> ()

// CIR: cir.func private @ext_short(!s16i {llvm.noundef, llvm.signext})
// CIR: cir.func private @ext_ptr(!cir.ptr<!void>)

// LLVM: define dso_local void @callers(
// LLVM:   call void @ext_short(i16 noundef signext %{{.*}})
// LLVM:   call void @ext_ptr(ptr %{{.*}})
// LLVM: declare void @ext_short(i16 noundef signext)
// LLVM: declare void @ext_ptr(ptr)

// NONOUNDEF: call void @ext_short(i16 signext %{{.*}})
// NONOUNDEF: declare void @ext_short(i16 signext)
