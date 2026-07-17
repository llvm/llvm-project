// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

enum BoolEnum : bool { False, True };

BoolEnum loadEnum(BoolEnum *p) { return *p; }

// CIR-LABEL: cir.func{{.*}} @_Z8loadEnumP8BoolEnum(%arg0: !cir.ptr<!cir.bool>
// CIR-SAME:    -> (!cir.bool
// CIR:         %[[P:.*]] = cir.alloca {{.*}} : !cir.ptr<!cir.ptr<!cir.bool>>
// CIR:         %[[PV:.*]] = cir.load deref {{.*}}%[[P]] : !cir.ptr<!cir.ptr<!cir.bool>>, !cir.ptr<!cir.bool>
// CIR:         %[[V:.*]] = cir.load {{.*}}%[[PV]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:         cir.return

// LLVM-LABEL: define dso_local noundef {{(zeroext )?}}i1 @_Z8loadEnumP8BoolEnum(ptr noundef %{{.*}})
// LLVM:         load i8, ptr %{{.*}}, align 1
// LLVM:         ret i1 %{{.*}}

void storeEnum(BoolEnum *p, BoolEnum v) { *p = v; }

// CIR-LABEL: cir.func{{.*}} @_Z9storeEnumP8BoolEnumS_(%arg0: !cir.ptr<!cir.bool>
// CIR-SAME:    %arg1: !cir.bool
// CIR:         %[[V:.*]] = cir.load {{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:         %[[P:.*]] = cir.load deref {{.*}} : !cir.ptr<!cir.ptr<!cir.bool>>, !cir.ptr<!cir.bool>
// CIR:         cir.store {{.*}}%[[V]], %[[P]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM-LABEL: define dso_local void @_Z9storeEnumP8BoolEnumS_(ptr noundef %{{.*}}, i1 noundef {{(zeroext )?}}%{{.*}})
// LLVM:         zext i1 %{{.*}} to i8
// LLVM:         store i8 %{{.*}}, ptr %{{.*}}, align 1
// LLVM:         load i8, ptr %{{.*}}, align 1
// LLVM:         store i8 %{{.*}}, ptr %{{.*}}, align 1

bool toBool(BoolEnum e) { return static_cast<bool>(e); }

// CIR-LABEL: cir.func{{.*}} @_Z6toBool8BoolEnum(%arg0: !cir.bool
// CIR-SAME:    -> (!cir.bool
// CIR:         cir.load {{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NOT:     cir.cast int_to_bool
// CIR:         cir.return

// LLVM-LABEL: define dso_local noundef {{(zeroext )?}}i1 @_Z6toBool8BoolEnum(i1 noundef {{(zeroext )?}}%{{.*}})
// LLVM:         ret i1 %{{.*}}
