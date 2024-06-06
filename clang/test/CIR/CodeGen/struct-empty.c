// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// CIR: ![[lock:.*]] = !cir.struct<struct "rwlock_t" {}>
// CIR: ![[fs_struct:.*]] = !cir.struct<struct "fs_struct" {!cir.struct<struct "rwlock_t" {}>, !cir.int<s, 32>}

typedef struct { } rwlock_t;
struct fs_struct { rwlock_t lock; int umask; };
void __copy_fs_struct(struct fs_struct *fs) { fs->lock = (rwlock_t) { }; }

// CIR-LABEL: __copy_fs_struct
// CIR:   %[[VAL_1:.*]] = cir.alloca !cir.ptr<![[fs_struct]]>, !cir.ptr<!cir.ptr<![[fs_struct]]>>, ["fs", init] {alignment = 8 : i64}
// CIR:   %[[VAL_2:.*]] = cir.alloca ![[lock]], !cir.ptr<![[lock]]>, [".compoundliteral"] {alignment = 1 : i64}
// CIR:   cir.store {{.*}}, %[[VAL_1]] : !cir.ptr<![[fs_struct]]>, !cir.ptr<!cir.ptr<![[fs_struct]]>>
// CIR:   %[[VAL_3:.*]] = cir.load %[[VAL_1]] : !cir.ptr<!cir.ptr<![[fs_struct]]>>, !cir.ptr<![[fs_struct]]>
// CIR:   %[[VAL_4:.*]] = cir.get_member %[[VAL_3]][0] {name = "lock"} : !cir.ptr<![[fs_struct]]> -> !cir.ptr<![[lock]]>
// CIR:   cir.copy %[[VAL_2]] to %[[VAL_4]] : !cir.ptr<![[lock]]>

// LLVM-LABEL: __copy_fs_struct
// LLVM:  %[[VAL_5:.*]] = getelementptr {{.*}}, {{.*}}, i32 0, i32 0
// LLVM:  call void @llvm.memcpy.p0.p0.i32(ptr %[[VAL_5]], ptr {{.*}}, i32 0, i1 false)