// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

enum BoolEnum : bool { False, True };

BoolEnum loadEnum(BoolEnum *p) { return *p; }

// CIR-LABEL: cir.func{{.*}} @_Z8loadEnumP8BoolEnum(%arg0: !cir.ptr<!cir.bool>
// CIR-SAME:    -> (!cir.bool
// CIR:         %[[P:.*]] = cir.alloca {{.*}} : !cir.ptr<!cir.ptr<!cir.bool>>
// CIR:         %[[PV:.*]] = cir.load deref {{.*}}%[[P]] : !cir.ptr<!cir.ptr<!cir.bool>>, !cir.ptr<!cir.bool>
// CIR:         %[[V:.*]] = cir.load {{.*}}%[[PV]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:         cir.return

// LLVM-LABEL: define dso_local noundef i1 @_Z8loadEnumP8BoolEnum(ptr noundef %0)
// LLVM:         %[[PTR:.*]] = load ptr, ptr
// LLVM:         %[[MEM:.*]] = load i8, ptr %[[PTR]], align 1
// LLVM:         %[[BOOL:.*]] = trunc i8 %[[MEM]] to i1

// OGCG-LABEL: define dso_local noundef zeroext i1 @_Z8loadEnumP8BoolEnum(ptr noundef %p)
// OGCG:         %[[MEM:.*]] = load i8, ptr {{.*}}, align 1
// OGCG:         %[[BOOL:.*]] = icmp ne i8 %[[MEM]], 0
// OGCG:         ret i1 %[[BOOL]]

void storeEnum(BoolEnum *p, BoolEnum v) { *p = v; }

// CIR-LABEL: cir.func{{.*}} @_Z9storeEnumP8BoolEnumS_(%arg0: !cir.ptr<!cir.bool>
// CIR-SAME:    %arg1: !cir.bool
// CIR:         %[[V:.*]] = cir.load {{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:         %[[P:.*]] = cir.load deref {{.*}} : !cir.ptr<!cir.ptr<!cir.bool>>, !cir.ptr<!cir.bool>
// CIR:         cir.store {{.*}}%[[V]], %[[P]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM-LABEL: define dso_local void @_Z9storeEnumP8BoolEnumS_(ptr noundef %0, i1 noundef %1)
// LLVM:         %[[Z:.*]] = zext i1 %1 to i8
// LLVM:         store i8 %[[Z]], ptr
// LLVM:         %[[LD:.*]] = load i8, ptr
// LLVM:         %[[TR:.*]] = trunc i8 %[[LD]] to i1
// LLVM:         %[[ZB:.*]] = zext i1 %[[TR]] to i8
// LLVM:         store i8 %[[ZB]], ptr

// OGCG-LABEL: define dso_local void @_Z9storeEnumP8BoolEnumS_(ptr noundef %p, i1 noundef zeroext %v)
// OGCG:         %[[SV:.*]] = zext i1 %v to i8
// OGCG:         store i8 %[[SV]], ptr %v.addr
// OGCG:         %[[LV:.*]] = icmp ne i8 {{.*}}, 0
// OGCG:         %[[SV1:.*]] = zext i1 %[[LV]] to i8
// OGCG:         store i8 %[[SV1]], ptr
