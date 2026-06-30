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

// A scoped enum with a boolean underlying type is compared directly (no
// integral promotion), so cir.cmp must accept !cir.bool operands.
enum class ScopedBoolEnum : bool { No, Yes };

bool eqEnum(ScopedBoolEnum a, ScopedBoolEnum b) { return a == b; }

// CIR-LABEL: cir.func{{.*}} @_Z6eqEnum14ScopedBoolEnumS_
// CIR:         cir.cmp eq %{{.*}}, %{{.*}} : !cir.bool

// LLVM-LABEL: define dso_local noundef {{(zeroext )?}}i1 @_Z6eqEnum14ScopedBoolEnumS_
// LLVM:         icmp eq i1 %{{.*}}, %{{.*}}

bool neEnum(ScopedBoolEnum a, ScopedBoolEnum b) { return a != b; }

// CIR-LABEL: cir.func{{.*}} @_Z6neEnum14ScopedBoolEnumS_
// CIR:         cir.cmp ne %{{.*}}, %{{.*}} : !cir.bool

// LLVM-LABEL: define dso_local noundef {{(zeroext )?}}i1 @_Z6neEnum14ScopedBoolEnumS_
// LLVM:         icmp ne i1 %{{.*}}, %{{.*}}

bool ltEnum(ScopedBoolEnum a, ScopedBoolEnum b) { return a < b; }

// CIR-LABEL: cir.func{{.*}} @_Z6ltEnum14ScopedBoolEnumS_
// CIR:         cir.cmp lt %{{.*}}, %{{.*}} : !cir.bool

// LLVM-LABEL: define dso_local noundef {{(zeroext )?}}i1 @_Z6ltEnum14ScopedBoolEnumS_
// LLVM:         icmp ult i1 %{{.*}}, %{{.*}}
