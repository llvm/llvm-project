// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mlong-double-64 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR -DLDTY=cir.double
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mlong-double-64 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM,LLVMCIR -DLDTY=double -DONE=1 -DNEGONE=-1
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mlong-double-64 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,OGCG -DLDTY=double -DONE=1 -DNEGONE=-1

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mlong-double-80 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR -DLDTY=cir.f80
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mlong-double-80 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM,LLVMCIR -DLDTY=x86_fp80 -DONE=0xK3FFF8 -DNEGONE=0xKBFFF8
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mlong-double-80 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,OGCG -DLDTY=x86_fp80 -DONE=0xK3FFF8 -DNEGONE=0xKBFFF8

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mlong-double-128 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR -DLDTY=cir.f128
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mlong-double-128 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM,LLVMCIR -DLDTY=fp128 -DONE=0xL00000000000000003FFF -DNEGONE=0xL0000000000000000BFFF
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mlong-double-128 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,OGCG -DLDTY=fp128 -DONE=0xL00000000000000003FFF -DNEGONE=0xL0000000000000000BFFF

extern "C" long double do_pre_inc(long double d) {
  // CIR-LABEL: @do_pre_inc(
  // CIR: %[[ARG_ALLOCA:.*]] = cir.alloca !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>, ["d", init]
  // CIR: %[[RET_ALLOCA:.*]] = cir.alloca !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>
  //
  // LLVM-LABEL: @do_pre_inc(
  // LLVM: %[[ARG_ALLOCA:.*]] = alloca [[LDTY]]
  // LLVMCIR: %[[RET_ALLOCA:.*]] = alloca [[LDTY]]

  return ++d;
  // CIR: %[[ARG_LOAD:.*]]  = cir.load {{.*}}%[[ARG_ALLOCA]] : !cir.ptr<!cir.long_double<![[LDTY]]>>, !cir.long_double<![[LDTY]]>
  // CIR: %[[ARG_INC:.*]] = cir.inc %[[ARG_LOAD]]
  // CIR: cir.store{{.*}} %[[ARG_INC]], %[[ARG_ALLOCA]] : !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>
  // CIR: cir.store %[[ARG_INC]], %[[RET_ALLOCA]] : !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>
  // CIR: %[[LOAD_RET:.*]] = cir.load %[[RET_ALLOCA]]
  // CIR: cir.return %[[LOAD_RET]] : !cir.long_double<![[LDTY]]>
  //
  // LLVM: %[[ARG_LOAD:.*]] = load [[LDTY]], ptr %[[ARG_ALLOCA]]
  // LLVMCIR: %[[ARG_INC:.*]] = fadd [[LDTY]] [[ONE]]{{.*}}, %[[ARG_LOAD]]
  // OGCG: %[[ARG_INC:.*]] = fadd [[LDTY]] %[[ARG_LOAD]], [[ONE]]{{.*}}
  // LLVM: store [[LDTY]] %[[ARG_INC]], ptr %[[ARG_ALLOCA]]
  // LLVMCIR: store [[LDTY]] %[[ARG_INC]], ptr %[[RET_ALLOCA]]
  // LLVMCIR: %[[LOAD_RET:.*]] = load [[LDTY]], ptr %[[RET_ALLOCA]]
  // LLVMCIR: ret [[LDTY]] %[[LOAD_RET]]
  // OGCG: ret [[LDTY]] %[[ARG_INC]]
}
extern "C" long double do_post_inc(long double d) {
  // CIR-LABEL: @do_post_inc(
  // CIR: %[[ARG_ALLOCA:.*]] = cir.alloca !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>, ["d", init]
  // CIR: %[[RET_ALLOCA:.*]] = cir.alloca !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>
  //
  // LLVM-LABEL: @do_post_inc(
  // LLVM: %[[ARG_ALLOCA:.*]] = alloca [[LDTY]]
  // LLVMCIR: %[[RET_ALLOCA:.*]] = alloca [[LDTY]]

  return d++;
  // CIR: %[[ARG_LOAD:.*]]  = cir.load {{.*}}%[[ARG_ALLOCA]] : !cir.ptr<!cir.long_double<![[LDTY]]>>, !cir.long_double<![[LDTY]]>
  // CIR: %[[ARG_INC:.*]] = cir.inc %[[ARG_LOAD]]
  // CIR: cir.store{{.*}} %[[ARG_INC]], %[[ARG_ALLOCA]] : !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>
  // CIR: cir.store %[[ARG_LOAD]], %[[RET_ALLOCA]] : !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>
  // CIR: %[[LOAD_RET:.*]] = cir.load %[[RET_ALLOCA]]
  // CIR: cir.return %[[LOAD_RET]] : !cir.long_double<![[LDTY]]>
  //
  // LLVM: %[[ARG_LOAD:.*]] = load [[LDTY]], ptr %[[ARG_ALLOCA]]
  // LLVMCIR: %[[ARG_INC:.*]] = fadd [[LDTY]] [[ONE]]{{.*}}, %[[ARG_LOAD]]
  // OGCG: %[[ARG_INC:.*]] = fadd [[LDTY]] %[[ARG_LOAD]], [[ONE]]{{.*}}
  // LLVM: store [[LDTY]] %[[ARG_INC]], ptr %[[ARG_ALLOCA]]
  // LLVMCIR: store [[LDTY]] %[[ARG_LOAD]], ptr %[[RET_ALLOCA]]
  // LLVMCIR: %[[LOAD_RET:.*]] = load [[LDTY]], ptr %[[RET_ALLOCA]]
  // LLVMCIR: ret [[LDTY]] %[[LOAD_RET]]
  // OGCG: ret [[LDTY]] %[[ARG_LOAD]]
}

extern "C" long double do_pre_dec(long double d) {
  // CIR-LABEL: @do_pre_dec(
  // CIR: %[[ARG_ALLOCA:.*]] = cir.alloca !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>, ["d", init]
  // CIR: %[[RET_ALLOCA:.*]] = cir.alloca !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>
  //
  // LLVM-LABEL: @do_pre_dec(
  // LLVM: %[[ARG_ALLOCA:.*]] = alloca [[LDTY]]
  // LLVMCIR: %[[RET_ALLOCA:.*]] = alloca [[LDTY]]

  return --d;
  // CIR: %[[ARG_LOAD:.*]]  = cir.load {{.*}}%[[ARG_ALLOCA]] : !cir.ptr<!cir.long_double<![[LDTY]]>>, !cir.long_double<![[LDTY]]>
  // CIR: %[[ARG_DEC:.*]] = cir.dec %[[ARG_LOAD]]
  // CIR: cir.store{{.*}} %[[ARG_DEC]], %[[ARG_ALLOCA]] : !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>
  // CIR: cir.store %[[ARG_DEC]], %[[RET_ALLOCA]] : !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>
  // CIR: %[[LOAD_RET:.*]] = cir.load %[[RET_ALLOCA]]
  // CIR: cir.return %[[LOAD_RET]] : !cir.long_double<![[LDTY]]>
  //
  // LLVM: %[[ARG_LOAD:.*]] = load [[LDTY]], ptr %[[ARG_ALLOCA]]
  // LLVMCIR: %[[ARG_DEC:.*]] = fadd [[LDTY]] [[NEGONE]]{{.*}}, %[[ARG_LOAD]]
  // OGCG: %[[ARG_DEC:.*]] = fadd [[LDTY]] %[[ARG_LOAD]], [[NEGONE]]{{.*}}
  // LLVM: store [[LDTY]] %[[ARG_DEC]], ptr %[[ARG_ALLOCA]]
  // LLVMCIR: store [[LDTY]] %[[ARG_DEC]], ptr %[[RET_ALLOCA]]
  // LLVMCIR: %[[LOAD_RET:.*]] = load [[LDTY]], ptr %[[RET_ALLOCA]]
  // LLVMCIR: ret [[LDTY]] %[[LOAD_RET]]
  // OGCG: ret [[LDTY]] %[[ARG_DEC]]
}
extern "C" long double do_post_dec(long double d) {
  // CIR-LABEL: @do_post_dec(
  // CIR: %[[ARG_ALLOCA:.*]] = cir.alloca !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>, ["d", init]
  // CIR: %[[RET_ALLOCA:.*]] = cir.alloca !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>
  //
  // LLVM-LABEL: @do_post_dec(
  // LLVM: %[[ARG_ALLOCA:.*]] = alloca [[LDTY]]
  // LLVMCIR: %[[RET_ALLOCA:.*]] = alloca [[LDTY]]

  return d--;
  // CIR: %[[ARG_LOAD:.*]]  = cir.load {{.*}}%[[ARG_ALLOCA]] : !cir.ptr<!cir.long_double<![[LDTY]]>>, !cir.long_double<![[LDTY]]>
  // CIR: %[[ARG_DEC:.*]] = cir.dec %[[ARG_LOAD]]
  // CIR: cir.store{{.*}} %[[ARG_DEC]], %[[ARG_ALLOCA]] : !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>
  // CIR: cir.store %[[ARG_LOAD]], %[[RET_ALLOCA]] : !cir.long_double<![[LDTY]]>, !cir.ptr<!cir.long_double<![[LDTY]]>>
  // CIR: %[[LOAD_RET:.*]] = cir.load %[[RET_ALLOCA]]
  // CIR: cir.return %[[LOAD_RET]] : !cir.long_double<![[LDTY]]>
  //
  // LLVM: %[[ARG_LOAD:.*]] = load [[LDTY]], ptr %[[ARG_ALLOCA]]
  // LLVMCIR: %[[ARG_DEC:.*]] = fadd [[LDTY]] [[NEGONE]]{{.*}}, %[[ARG_LOAD]]
  // OGCG: %[[ARG_DEC:.*]] = fadd [[LDTY]] %[[ARG_LOAD]], [[NEGONE]]{{.*}}
  // LLVM: store [[LDTY]] %[[ARG_DEC]], ptr %[[ARG_ALLOCA]]
  // LLVMCIR: store [[LDTY]] %[[ARG_LOAD]], ptr %[[RET_ALLOCA]]
  // LLVMCIR: %[[LOAD_RET:.*]] = load [[LDTY]], ptr %[[RET_ALLOCA]]
  // LLVMCIR: ret [[LDTY]] %[[LOAD_RET]]
  // OGCG: ret [[LDTY]] %[[ARG_LOAD]]
}
