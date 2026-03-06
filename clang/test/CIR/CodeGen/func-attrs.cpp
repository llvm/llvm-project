// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -O1 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s
//
// Test -O0 implicit optnone:
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -O0 -fclangir -emit-cir %s -o %t-o0.cir
// RUN: FileCheck --check-prefix=O0CIR --input-file=%t-o0.cir %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -O0 -fclangir -emit-llvm %s -o %t-o0.ll
// RUN: FileCheck --check-prefix=O0LLVM --input-file=%t-o0.ll %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -O0 -emit-llvm %s -o %t-o0-ogcg.ll
// RUN: FileCheck --check-prefix=O0OGCG --input-file=%t-o0-ogcg.ll %s

extern "C" {

// CIR: cir.func no_inline dso_local @optnone_func()
// CIR-SAME: attributes {optimize_none}
// LLVM: define dso_local void @optnone_func() {{.*}}#[[OPTNONE_ATTR:[0-9]+]]
// OGCG: define dso_local void @optnone_func() {{.*}}#[[OPTNONE_ATTR:[0-9]+]]
__attribute__((optnone))
void optnone_func() {}

// CIR: cir.func dso_local @cold_func()
// CIR-SAME: attributes {cold, optsize}
// LLVM: define dso_local void @cold_func() {{.*}}#[[COLD_ATTR:[0-9]+]]
// OGCG: define dso_local void @cold_func() {{.*}}#[[COLD_ATTR:[0-9]+]]
__attribute__((cold))
void cold_func() {}

// CIR: cir.func dso_local @hot_func()
// CIR-SAME: attributes {hot}
// LLVM: define dso_local void @hot_func() {{.*}}#[[HOT_ATTR:[0-9]+]]
// OGCG: define dso_local void @hot_func() {{.*}}#[[HOT_ATTR:[0-9]+]]
__attribute__((hot))
void hot_func() {}

// CIR: cir.func dso_local @noduplicate_func()
// CIR-SAME: attributes {noduplicate}
// LLVM: define dso_local void @noduplicate_func() {{.*}}#[[ND_ATTR:[0-9]+]]
// OGCG: define dso_local void @noduplicate_func() {{.*}}#[[ND_ATTR:[0-9]+]]
__attribute__((noduplicate))
void noduplicate_func() {}

// CIR: cir.func dso_local @minsize_func()
// CIR-SAME: attributes {minsize}
// LLVM: define dso_local void @minsize_func() {{.*}}#[[MINSIZE_ATTR:[0-9]+]]
// OGCG: define dso_local void @minsize_func() {{.*}}#[[MINSIZE_ATTR:[0-9]+]]
__attribute__((minsize))
void minsize_func() {}

// optnone + cold: optnone wins, cold stays but optsize must NOT be added.
// CIR: cir.func no_inline dso_local @optnone_cold_func()
// CIR-SAME: attributes {cold, optimize_none}
// CIR-NOT: optsize
// LLVM: define dso_local void @optnone_cold_func() {{.*}}#[[OPTNONE_COLD_ATTR:[0-9]+]]
// OGCG: define dso_local void @optnone_cold_func() {{.*}}#[[OPTNONE_COLD_ATTR:[0-9]+]]
__attribute__((optnone, cold))
void optnone_cold_func() {}

// optnone + hot: hot stays (set by constructAttributeList), optnone+noinline added.
// CIR: cir.func no_inline dso_local @optnone_hot_func()
// CIR-SAME: attributes {hot, optimize_none}
// LLVM: define dso_local void @optnone_hot_func() {{.*}}#[[OPTNONE_HOT_ATTR:[0-9]+]]
// OGCG: define dso_local void @optnone_hot_func() {{.*}}#[[OPTNONE_HOT_ATTR:[0-9]+]]
__attribute__((optnone, hot))
void optnone_hot_func() {}

// always_inline: gets alwaysinline, no optnone even at -O0.
// CIR: cir.func always_inline dso_local @always_inline_func() {
// LLVM: define dso_local void @always_inline_func() {{.*}}#[[AI_ATTR:[0-9]+]]
// OGCG: define dso_local void @always_inline_func() {{.*}}#[[AI_ATTR:[0-9]+]]
__attribute__((always_inline))
void always_inline_func() {}

// -O0 implicit optnone: normal functions get optimize_none + noinline.
// O0CIR: cir.func no_inline dso_local @optnone_func()
// O0CIR-SAME: optimize_none
// O0CIR: cir.func no_inline dso_local @cold_func()
// O0CIR-SAME: optimize_none
// -O0 with hot: optnone takes precedence, hot stays.
// O0CIR: cir.func no_inline dso_local @hot_func()
// O0CIR-SAME: optimize_none
// -O0 with noduplicate: optnone takes precedence, noduplicate stays.
// O0CIR: cir.func no_inline dso_local @noduplicate_func()
// O0CIR-SAME: optimize_none
// minsize suppresses implicit optnone at -O0.
// O0CIR: cir.func no_inline dso_local @minsize_func() attributes {minsize} {
// always_inline suppresses implicit optnone at -O0.
// O0CIR: cir.func always_inline dso_local @always_inline_func() {
//
// O0LLVM: define dso_local void @optnone_func() {{.*}}#[[O0_OPTNONE:[0-9]+]]
// O0LLVM: define dso_local void @always_inline_func() {{.*}}#[[O0_AI:[0-9]+]]
// O0LLVM-DAG: attributes #[[O0_OPTNONE]] = {{{.*}}noinline{{.*}}optnone{{.*}}}
// O0LLVM-DAG: attributes #[[O0_AI]] = {{{.*}}alwaysinline{{.*}}}
//
// O0OGCG: define dso_local void @optnone_func() {{.*}}#[[O0_OPTNONE:[0-9]+]]
// O0OGCG: define dso_local void @always_inline_func() {{.*}}#[[O0_AI:[0-9]+]]
// O0OGCG-DAG: attributes #[[O0_OPTNONE]] = {{{.*}}noinline{{.*}}optnone{{.*}}}
// O0OGCG-DAG: attributes #[[O0_AI]] = {{{.*}}alwaysinline{{.*}}}

}

// LLVM-DAG: attributes #[[OPTNONE_ATTR]] = {{{.*}}noinline{{.*}}optnone{{.*}}}
// LLVM-DAG: attributes #[[COLD_ATTR]] = {{{.*}}cold{{.*}}optsize{{.*}}}
// LLVM-DAG: attributes #[[HOT_ATTR]] = {{{.*}}hot{{.*}}}
// LLVM-DAG: attributes #[[ND_ATTR]] = {{{.*}}noduplicate{{.*}}}
// LLVM-DAG: attributes #[[MINSIZE_ATTR]] = {{{.*}}minsize{{.*}}}
// LLVM-DAG: attributes #[[OPTNONE_COLD_ATTR]] = {{{.*}}cold{{.*}}noinline{{.*}}optnone{{.*}}}
// LLVM-DAG: attributes #[[OPTNONE_HOT_ATTR]] = {{{.*}}hot{{.*}}noinline{{.*}}optnone{{.*}}}
// LLVM-DAG: attributes #[[AI_ATTR]] = {{{.*}}alwaysinline{{.*}}}

// OGCG-DAG: attributes #[[OPTNONE_ATTR]] = {{{.*}}noinline{{.*}}optnone{{.*}}}
// OGCG-DAG: attributes #[[COLD_ATTR]] = {{{.*}}cold{{.*}}optsize{{.*}}}
// OGCG-DAG: attributes #[[HOT_ATTR]] = {{{.*}}hot{{.*}}}
// OGCG-DAG: attributes #[[ND_ATTR]] = {{{.*}}noduplicate{{.*}}}
// OGCG-DAG: attributes #[[MINSIZE_ATTR]] = {{{.*}}minsize{{.*}}}
// OGCG-DAG: attributes #[[OPTNONE_COLD_ATTR]] = {{{.*}}cold{{.*}}noinline{{.*}}optnone{{.*}}}
// OGCG-DAG: attributes #[[OPTNONE_HOT_ATTR]] = {{{.*}}hot{{.*}}noinline{{.*}}optnone{{.*}}}
// OGCG-DAG: attributes #[[AI_ATTR]] = {{{.*}}alwaysinline{{.*}}}
