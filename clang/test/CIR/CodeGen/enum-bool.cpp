// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

enum BoolEnum : bool { False, True };

BoolEnum loadEnum(BoolEnum *p) { return *p; }

// CIR-LABEL: cir.func{{.*}} @_Z8loadEnumP8BoolEnum(%arg0: !cir.ptr<!cir.bool>
// CIR-SAME:    -> (!cir.bool
// CIR:         %[[P:.*]] = cir.alloca {{.*}} : !cir.ptr<!cir.ptr<!cir.bool>>
// CIR:         %[[PV:.*]] = cir.load deref {{.*}}%[[P]] : !cir.ptr<!cir.ptr<!cir.bool>>, !cir.ptr<!cir.bool>
// CIR:         %[[V:.*]] = cir.load {{.*}}%[[PV]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:         cir.return

// LLVMCIR-LABEL: define dso_local noundef i1 @_Z8loadEnumP8BoolEnum(ptr noundef %{{.*}})
// OGCG-LABEL: define dso_local noundef zeroext i1 @_Z8loadEnumP8BoolEnum(ptr noundef %{{.*}})
// LLVM:         load i8, ptr %{{.*}}, align 1
// LLVM:         ret i1 %{{.*}}

void storeEnum(BoolEnum *p, BoolEnum v) { *p = v; }

// CIR-LABEL: cir.func{{.*}} @_Z9storeEnumP8BoolEnumS_(%arg0: !cir.ptr<!cir.bool>
// CIR-SAME:    %arg1: !cir.bool
// CIR:         %[[V:.*]] = cir.load {{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:         %[[P:.*]] = cir.load deref {{.*}} : !cir.ptr<!cir.ptr<!cir.bool>>, !cir.ptr<!cir.bool>
// CIR:         cir.store {{.*}}%[[V]], %[[P]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVMCIR-LABEL: define dso_local void @_Z9storeEnumP8BoolEnumS_(ptr noundef %{{.*}}, i1 noundef %{{.*}})
// OGCG-LABEL: define dso_local void @_Z9storeEnumP8BoolEnumS_(ptr noundef %{{.*}}, i1 noundef zeroext %{{.*}})
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

// LLVMCIR-LABEL: define dso_local noundef i1 @_Z6toBool8BoolEnum(i1 noundef %{{.*}})
// OGCG-LABEL: define dso_local noundef zeroext i1 @_Z6toBool8BoolEnum(i1 noundef zeroext %{{.*}})
// LLVM:         ret i1 %{{.*}}

// An unscoped enum is integer-promoted before the comparison, so cir.cmp sees
// !s32i operands.
bool ltUnscopedEnum(BoolEnum a, BoolEnum b) { return a < b; }

// CIR-LABEL: cir.func{{.*}} @_Z14ltUnscopedEnum8BoolEnumS_
// CIR:         %[[A:.*]] = cir.cast bool_to_int %{{.*}} : !cir.bool -> !s32i
// CIR:         %[[B:.*]] = cir.cast bool_to_int %{{.*}} : !cir.bool -> !s32i
// CIR:         cir.cmp lt %[[A]], %[[B]] : !s32i

// LLVMCIR-LABEL: define dso_local noundef i1 @_Z14ltUnscopedEnum8BoolEnumS_(i1 noundef %{{.*}}, i1 noundef %{{.*}})
// OGCG-LABEL: define dso_local noundef zeroext i1 @_Z14ltUnscopedEnum8BoolEnumS_(i1 noundef zeroext %{{.*}}, i1 noundef zeroext %{{.*}})
// LLVM:         zext i1 %{{.*}} to i32
// LLVM:         zext i1 %{{.*}} to i32
// LLVM:         icmp slt i32 %{{.*}}, %{{.*}}

// Plain bool is promoted the same way.
bool ltPlainBool(bool a, bool b) { return a < b; }

// CIR-LABEL: cir.func{{.*}} @_Z11ltPlainBoolbb
// CIR:         %[[A:.*]] = cir.cast bool_to_int %{{.*}} : !cir.bool -> !s32i
// CIR:         %[[B:.*]] = cir.cast bool_to_int %{{.*}} : !cir.bool -> !s32i
// CIR:         cir.cmp lt %[[A]], %[[B]] : !s32i

// LLVMCIR-LABEL: define dso_local noundef i1 @_Z11ltPlainBoolbb(i1 noundef %{{.*}}, i1 noundef %{{.*}})
// OGCG-LABEL: define dso_local noundef zeroext i1 @_Z11ltPlainBoolbb(i1 noundef zeroext %{{.*}}, i1 noundef zeroext %{{.*}})
// LLVM:         zext i1 %{{.*}} to i32
// LLVM:         zext i1 %{{.*}} to i32
// LLVM:         icmp slt i32 %{{.*}}, %{{.*}}

// A scoped enum with a boolean underlying type is compared directly (no
// integral promotion), so cir.cmp must accept !cir.bool operands.
enum class ScopedBoolEnum : bool { No, Yes };

bool eqEnum(ScopedBoolEnum a, ScopedBoolEnum b) { return a == b; }

// CIR-LABEL: cir.func{{.*}} @_Z6eqEnum14ScopedBoolEnumS_
// CIR:         cir.cmp eq %{{.*}}, %{{.*}} : !cir.bool

// LLVMCIR-LABEL: define dso_local noundef i1 @_Z6eqEnum14ScopedBoolEnumS_(i1 noundef %{{.*}}, i1 noundef %{{.*}})
// OGCG-LABEL: define dso_local noundef zeroext i1 @_Z6eqEnum14ScopedBoolEnumS_(i1 noundef zeroext %{{.*}}, i1 noundef zeroext %{{.*}})
// LLVM:         icmp eq i1 %{{.*}}, %{{.*}}

bool neEnum(ScopedBoolEnum a, ScopedBoolEnum b) { return a != b; }

// CIR-LABEL: cir.func{{.*}} @_Z6neEnum14ScopedBoolEnumS_
// CIR:         cir.cmp ne %{{.*}}, %{{.*}} : !cir.bool

// LLVMCIR-LABEL: define dso_local noundef i1 @_Z6neEnum14ScopedBoolEnumS_(i1 noundef %{{.*}}, i1 noundef %{{.*}})
// OGCG-LABEL: define dso_local noundef zeroext i1 @_Z6neEnum14ScopedBoolEnumS_(i1 noundef zeroext %{{.*}}, i1 noundef zeroext %{{.*}})
// LLVM:         icmp ne i1 %{{.*}}, %{{.*}}

bool ltEnum(ScopedBoolEnum a, ScopedBoolEnum b) { return a < b; }

// CIR-LABEL: cir.func{{.*}} @_Z6ltEnum14ScopedBoolEnumS_
// CIR:         cir.cmp lt %{{.*}}, %{{.*}} : !cir.bool

// LLVMCIR-LABEL: define dso_local noundef i1 @_Z6ltEnum14ScopedBoolEnumS_(i1 noundef %{{.*}}, i1 noundef %{{.*}})
// OGCG-LABEL: define dso_local noundef zeroext i1 @_Z6ltEnum14ScopedBoolEnumS_(i1 noundef zeroext %{{.*}}, i1 noundef zeroext %{{.*}})
// LLVM:         icmp ult i1 %{{.*}}, %{{.*}}

bool leEnum(ScopedBoolEnum a, ScopedBoolEnum b) { return a <= b; }

// CIR-LABEL: cir.func{{.*}} @_Z6leEnum14ScopedBoolEnumS_
// CIR:         cir.cmp le %{{.*}}, %{{.*}} : !cir.bool

// LLVMCIR-LABEL: define dso_local noundef i1 @_Z6leEnum14ScopedBoolEnumS_(i1 noundef %{{.*}}, i1 noundef %{{.*}})
// OGCG-LABEL: define dso_local noundef zeroext i1 @_Z6leEnum14ScopedBoolEnumS_(i1 noundef zeroext %{{.*}}, i1 noundef zeroext %{{.*}})
// LLVM:         icmp ule i1 %{{.*}}, %{{.*}}

bool gtEnum(ScopedBoolEnum a, ScopedBoolEnum b) { return a > b; }

// CIR-LABEL: cir.func{{.*}} @_Z6gtEnum14ScopedBoolEnumS_
// CIR:         cir.cmp gt %{{.*}}, %{{.*}} : !cir.bool

// LLVMCIR-LABEL: define dso_local noundef i1 @_Z6gtEnum14ScopedBoolEnumS_(i1 noundef %{{.*}}, i1 noundef %{{.*}})
// OGCG-LABEL: define dso_local noundef zeroext i1 @_Z6gtEnum14ScopedBoolEnumS_(i1 noundef zeroext %{{.*}}, i1 noundef zeroext %{{.*}})
// LLVM:         icmp ugt i1 %{{.*}}, %{{.*}}

bool geEnum(ScopedBoolEnum a, ScopedBoolEnum b) { return a >= b; }

// CIR-LABEL: cir.func{{.*}} @_Z6geEnum14ScopedBoolEnumS_
// CIR:         cir.cmp ge %{{.*}}, %{{.*}} : !cir.bool

// LLVMCIR-LABEL: define dso_local noundef i1 @_Z6geEnum14ScopedBoolEnumS_(i1 noundef %{{.*}}, i1 noundef %{{.*}})
// OGCG-LABEL: define dso_local noundef zeroext i1 @_Z6geEnum14ScopedBoolEnumS_(i1 noundef zeroext %{{.*}}, i1 noundef zeroext %{{.*}})
// LLVM:         icmp uge i1 %{{.*}}, %{{.*}}
