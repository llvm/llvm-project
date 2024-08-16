// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK0 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix CK0 --check-prefix CK0-64 %s
// RUN: %clang_cc1 -DCK0 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK0 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix CK0 --check-prefix CK0-64 %s
// RUN: %clang_cc1 -DCK0 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix CK0 --check-prefix CK0-32 %s
// RUN: %clang_cc1 -DCK0 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK0 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix CK0 --check-prefix CK0-32 %s

// RUN: %clang_cc1 -DCK0 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK0 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK0 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK0 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK0 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK0 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s

#ifdef CK0
// Mapper function code generation and runtime interface.

// CK0: [[IDENT_T:%.+]] = type { i32, i32, i32, i32, ptr }
// CK0: [[ENTRY:%.+]] = type { ptr, ptr, i[[SZ:32|64]], i32, i32 }
// CK0: [[ANON_T:%.+]] = type { ptr }
// CK0: [[ANON_T_0:%.+]] = type { ptr }
// CK0: [[KMP_TASK_T_WITH_PRIVATES:%.+]] = type { [[KMP_TASK_T:%[^,]+]], [[KMP_PRIVATES_T:%.+]] }
// CK0: [[KMP_TASK_T]] = type { ptr, ptr, i32, %{{[^,]+}}, %{{[^,]+}} }
// CK0-32: [[KMP_PRIVATES_T]] = type { [1 x i64], [1 x ptr], [1 x ptr], [1 x ptr] }
// CK0-64: [[KMP_PRIVATES_T]] = type { [1 x ptr], [1 x ptr], [1 x i64], [1 x ptr] }
// CK0: [[KMP_TASK_T_WITH_PRIVATES_1:%.+]] = type { [[KMP_TASK_T]], [[KMP_PRIVATES_T_2:%.+]] }
// CK0-32: [[KMP_PRIVATES_T_2]] = type { [1 x i64], [1 x ptr], [1 x ptr], [1 x ptr] }
// CK0-64: [[KMP_PRIVATES_T_2]] = type { [1 x ptr], [1 x ptr], [1 x i64], [1 x ptr] }
// CK0: [[KMP_TASK_T_WITH_PRIVATES_4:%.+]] = type { [[KMP_TASK_T]], [[KMP_PRIVATES_T_5:%.+]] }
// CK0-32: [[KMP_PRIVATES_T_5]] = type { [1 x i64], [1 x ptr], [1 x ptr], [1 x ptr] }
// CK0-64: [[KMP_PRIVATES_T_5]] = type { [1 x ptr], [1 x ptr], [1 x i64], [1 x ptr] }
// CK0: [[KMP_TASK_T_WITH_PRIVATES_7:%.+]] = type { [[KMP_TASK_T]], [[KMP_PRIVATES_T_8:%.+]] }
// CK0-32: [[KMP_PRIVATES_T_8]] = type { [1 x i64], [1 x ptr], [1 x ptr], [1 x ptr] }
// CK0-64: [[KMP_PRIVATES_T_8]] = type { [1 x ptr], [1 x ptr], [1 x i64], [1 x ptr] }
// CK0: [[KMP_TASK_T_WITH_PRIVATES_10:%.+]] = type { [[KMP_TASK_T]], [[KMP_PRIVATES_T_11:%.+]] }
// CK0-32: [[KMP_PRIVATES_T_11]] = type { [1 x i64], [1 x ptr], [1 x ptr], [1 x ptr] }
// CK0-64: [[KMP_PRIVATES_T_11]] = type { [1 x ptr], [1 x ptr], [1 x i64], [1 x ptr] }

// CK0-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}.region_id = weak constant i8 0
// CK0-64: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 35]
// CK0-64: [[NWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[NWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[NWTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 35]
// CK0-64: [[TEAMSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[TEAMSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[TEAMTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 33]
// CK0-64: [[TEAMNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[TEAMNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[TEAMNWTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 33]
// CK0-64: [[EDSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[EDSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[EDTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 1]
// CK0-64: [[EDNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[EDNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[EDNWTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 1]
// CK0-64: [[EXDSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[EXDSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[EXDTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 2]
// CK0-64: [[EXDNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[EXDNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[EXDNWTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 2]
// CK0-64: [[TSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[TSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[TTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 1]
// CK0-64: [[FSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[FSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[FTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 2]
// CK0-64: [[FNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[FNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[FNWTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 2]

class C {
public:
  int a;
  double *b;
};

#pragma omp declare mapper(id: C s) map(s.a, s.b[0:2])

// CK0: define {{.*}}void [[MPRFUNC:@[.]omp_mapper[.].*C[.]id]](ptr{{.*}}, ptr{{.*}}, ptr{{.*}}, i64{{.*}}, i64{{.*}}, ptr{{.*}})
// CK0: store ptr %{{[^,]+}}, ptr [[HANDLEADDR:%[^,]+]]
// CK0: store ptr %{{[^,]+}}, ptr [[BPTRADDR:%[^,]+]]
// CK0: store ptr %{{[^,]+}}, ptr [[VPTRADDR:%[^,]+]]
// CK0: store i64 %{{[^,]+}}, ptr [[SIZEADDR:%[^,]+]]
// CK0: store i64 %{{[^,]+}}, ptr [[TYPEADDR:%[^,]+]]
// CK0-DAG: [[BYTESIZE:%.+]] = load i64, ptr [[SIZEADDR]]
// CK0-64-DAG: [[SIZE:%.+]] = udiv exact i64 [[BYTESIZE]], 16
// CK0-32-DAG: [[SIZE:%.+]] = udiv exact i64 [[BYTESIZE]], 8
// CK0-DAG: [[TYPE:%.+]] = load i64, ptr [[TYPEADDR]]
// CK0-DAG: [[HANDLE:%.+]] = load ptr, ptr [[HANDLEADDR]]
// CK0-DAG: [[BPTR:%.+]] = load ptr, ptr [[BPTRADDR]]
// CK0-DAG: [[BEGIN:%.+]] = load ptr, ptr [[VPTRADDR]]
// CK0-DAG: [[ISARRAY:%.+]] = icmp sgt i64 [[SIZE]], 1
// CK0-DAG: [[PTREND:%.+]] = getelementptr %class.C, ptr [[BEGIN]], i64 [[SIZE]]
// CK0-DAG: [[PTRSNE:%.+]] = icmp ne ptr [[BPTR]], [[BEGIN]]
// CK0-DAG: [[PTRANDOBJ:%.+]] = and i64 [[TYPE]], 16
// CK0-DAG: [[ISPTRANDOBJ:%.+]] = icmp ne i64 [[PTRANDOBJ]], 0
// CK0-DAG: [[CMPA:%.+]] = and i1 [[PTRSNE]], [[ISPTRANDOBJ]]
// CK0-DAG: [[CMP:%.+]] = or i1 [[ISARRAY]], [[CMPA]]
// CK0-DAG: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK0-DAG: [[ISNOTDEL:%.+]] = icmp eq i64 [[TYPEDEL]], 0
// CK0-DAG: [[CMP1:%.+]] = and i1 [[CMP]], [[ISNOTDEL]]
// CK0: br i1 [[CMP1]], label %[[INIT:[^,]+]], label %[[LHEAD:[^,]+]]
// CK0: [[INIT]]
// CK0-64-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 16
// CK0-32-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 8

// Remove movement mappings and mark as implicit
// CK0-DAG: [[ITYPE:%.+]] = and i64 [[TYPE]], -4
// CK0-DAG: [[ITYPE1:%.+]] = or i64 [[ITYPE]], 512
// CK0: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[BPTR]], ptr [[BEGIN]], i64 [[ARRSIZE]], i64 [[ITYPE1]], {{.*}})
// CK0: br label %[[LHEAD:[^,]+]]

// CK0: [[LHEAD]]
// CK0: [[ISEMPTY:%.+]] = icmp eq ptr [[BEGIN]], [[PTREND]]
// CK0: br i1 [[ISEMPTY]], label %[[DONE:[^,]+]], label %[[LBODY:[^,]+]]
// CK0: [[LBODY]]
// CK0: [[PTR:%.+]] = phi ptr [ [[BEGIN]], %{{.+}} ], [ [[PTRNEXT:%.+]], %[[LCORRECT:[^,]+]] ]
// CK0-DAG: [[ABEGIN:%.+]] = getelementptr inbounds nuw %class.C, ptr [[PTR]], i32 0, i32 0
// CK0-DAG: [[BBEGIN:%.+]] = getelementptr inbounds nuw %class.C, ptr [[PTR]], i32 0, i32 1
// CK0-DAG: [[BBEGIN2:%.+]] = getelementptr inbounds nuw %class.C, ptr [[PTR]], i32 0, i32 1
// CK0-DAG: [[BARRBEGIN:%.+]] = load ptr, ptr [[BBEGIN2]]
// CK0-DAG: [[BARRBEGINGEP:%.+]] = getelementptr inbounds double, ptr [[BARRBEGIN]], i[[sz:64|32]] 0
// CK0-DAG: [[BEND:%.+]] = getelementptr ptr, ptr [[BBEGIN]], i32 1
// CK0-DAG: [[ABEGINI:%.+]] = ptrtoint ptr [[ABEGIN]] to i64
// CK0-DAG: [[BENDI:%.+]] = ptrtoint ptr [[BEND]] to i64
// CK0-DAG: [[CSIZE:%.+]] = sub i64 [[BENDI]], [[ABEGINI]]
// CK0-DAG: [[CUSIZE:%.+]] = sdiv exact i64 [[CSIZE]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CK0-DAG: [[PRESIZE:%.+]] = call i64 @__tgt_mapper_num_components(ptr [[HANDLE]])
// CK0-DAG: [[SHIPRESIZE:%.+]] = shl i64 [[PRESIZE]], 48
// CK0-DAG: [[MEMBERTYPE:%.+]] = add nuw i64 0, [[SHIPRESIZE]]
// CK0-DAG: [[TYPETF:%.+]] = and i64 [[TYPE]], 3
// CK0-DAG: [[ISALLOC:%.+]] = icmp eq i64 [[TYPETF]], 0
// CK0-DAG: br i1 [[ISALLOC]], label %[[ALLOC:[^,]+]], label %[[ALLOCELSE:[^,]+]]
// CK0-DAG: [[ALLOC]]
// CK0-DAG: [[ALLOCTYPE:%.+]] = and i64 [[MEMBERTYPE]], -4
// CK0-DAG: br label %[[TYEND:[^,]+]]
// CK0-DAG: [[ALLOCELSE]]
// CK0-DAG: [[ISTO:%.+]] = icmp eq i64 [[TYPETF]], 1
// CK0-DAG: br i1 [[ISTO]], label %[[TO:[^,]+]], label %[[TOELSE:[^,]+]]
// CK0-DAG: [[TO]]
// CK0-DAG: [[TOTYPE:%.+]] = and i64 [[MEMBERTYPE]], -3
// CK0-DAG: br label %[[TYEND]]
// CK0-DAG: [[TOELSE]]
// CK0-DAG: [[ISFROM:%.+]] = icmp eq i64 [[TYPETF]], 2
// CK0-DAG: br i1 [[ISFROM]], label %[[FROM:[^,]+]], label %[[TYEND]]
// CK0-DAG: [[FROM]]
// CK0-DAG: [[FROMTYPE:%.+]] = and i64 [[MEMBERTYPE]], -2
// CK0-DAG: br label %[[TYEND]]
// CK0-DAG: [[TYEND]]
// CK0-DAG: [[PHITYPE0:%.+]] = phi i64 [ [[ALLOCTYPE]], %[[ALLOC]] ], [ [[TOTYPE]], %[[TO]] ], [ [[FROMTYPE]], %[[FROM]] ], [ [[MEMBERTYPE]], %[[TOELSE]] ]
// CK0: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[PTR]], ptr [[ABEGIN]], i64 [[CUSIZE]], i64 [[PHITYPE0]], {{.*}})
// 281474976710659 == 0x1,000,000,003
// CK0-DAG: [[MEMBERTYPE:%.+]] = add nuw i64 281474976710659, [[SHIPRESIZE]]
// CK0-DAG: [[TYPETF:%.+]] = and i64 [[TYPE]], 3
// CK0-DAG: [[ISALLOC:%.+]] = icmp eq i64 [[TYPETF]], 0
// CK0-DAG: br i1 [[ISALLOC]], label %[[ALLOC:[^,]+]], label %[[ALLOCELSE:[^,]+]]
// CK0-DAG: [[ALLOC]]
// CK0-DAG: [[ALLOCTYPE:%.+]] = and i64 [[MEMBERTYPE]], -4
// CK0-DAG: br label %[[TYEND:[^,]+]]
// CK0-DAG: [[ALLOCELSE]]
// CK0-DAG: [[ISTO:%.+]] = icmp eq i64 [[TYPETF]], 1
// CK0-DAG: br i1 [[ISTO]], label %[[TO:[^,]+]], label %[[TOELSE:[^,]+]]
// CK0-DAG: [[TO]]
// CK0-DAG: [[TOTYPE:%.+]] = and i64 [[MEMBERTYPE]], -3
// CK0-DAG: br label %[[TYEND]]
// CK0-DAG: [[TOELSE]]
// CK0-DAG: [[ISFROM:%.+]] = icmp eq i64 [[TYPETF]], 2
// CK0-DAG: br i1 [[ISFROM]], label %[[FROM:[^,]+]], label %[[TYEND]]
// CK0-DAG: [[FROM]]
// CK0-DAG: [[FROMTYPE:%.+]] = and i64 [[MEMBERTYPE]], -2
// CK0-DAG: br label %[[TYEND]]
// CK0-DAG: [[TYEND]]
// CK0-DAG: [[TYPE1:%.+]] = phi i64 [ [[ALLOCTYPE]], %[[ALLOC]] ], [ [[TOTYPE]], %[[TO]] ], [ [[FROMTYPE]], %[[FROM]] ], [ [[MEMBERTYPE]], %[[TOELSE]] ]
// CK0: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[PTR]], ptr [[ABEGIN]], i64 4, i64 [[TYPE1]], {{.*}})
// 281474976710675 == 0x1,000,000,013
// CK0-DAG: [[MEMBERTYPE:%.+]] = add nuw i64 281474976710675, [[SHIPRESIZE]]
// CK0-DAG: [[TYPETF:%.+]] = and i64 [[TYPE]], 3
// CK0-DAG: [[ISALLOC:%.+]] = icmp eq i64 [[TYPETF]], 0
// CK0-DAG: br i1 [[ISALLOC]], label %[[ALLOC:[^,]+]], label %[[ALLOCELSE:[^,]+]]
// CK0-DAG: [[ALLOC]]
// CK0-DAG: [[ALLOCTYPE:%.+]] = and i64 [[MEMBERTYPE]], -4
// CK0-DAG: br label %[[TYEND:[^,]+]]
// CK0-DAG: [[ALLOCELSE]]
// CK0-DAG: [[ISTO:%.+]] = icmp eq i64 [[TYPETF]], 1
// CK0-DAG: br i1 [[ISTO]], label %[[TO:[^,]+]], label %[[TOELSE:[^,]+]]
// CK0-DAG: [[TO]]
// CK0-DAG: [[TOTYPE:%.+]] = and i64 [[MEMBERTYPE]], -3
// CK0-DAG: br label %[[TYEND]]
// CK0-DAG: [[TOELSE]]
// CK0-DAG: [[ISFROM:%.+]] = icmp eq i64 [[TYPETF]], 2
// CK0-DAG: br i1 [[ISFROM]], label %[[FROM:[^,]+]], label %[[TYEND]]
// CK0-DAG: [[FROM]]
// CK0-DAG: [[FROMTYPE:%.+]] = and i64 [[MEMBERTYPE]], -2
// CK0-DAG: br label %[[TYEND]]
// CK0-DAG: [[TYEND]]
// CK0-DAG: [[TYPE2:%.+]] = phi i64 [ [[ALLOCTYPE]], %[[ALLOC]] ], [ [[TOTYPE]], %[[TO]] ], [ [[FROMTYPE]], %[[FROM]] ], [ [[MEMBERTYPE]], %[[TOELSE]] ]
// CK0: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[BBEGIN]], ptr [[BARRBEGINGEP]], i64 16, i64 [[TYPE2]], {{.*}})
// CK0: [[PTRNEXT]] = getelementptr %class.C, ptr [[PTR]], i32 1
// CK0: [[ISDONE:%.+]] = icmp eq ptr [[PTRNEXT]], [[PTREND]]
// CK0: br i1 [[ISDONE]], label %[[LEXIT:[^,]+]], label %[[LBODY]]

// CK0: [[LEXIT]]
// CK0: [[ISARRAY:%.+]] = icmp sgt i64 [[SIZE]], 1
// CK0: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK0: [[ISNOTDEL:%.+]] = icmp ne i64 [[TYPEDEL]], 0
// CK0: [[CMP1:%.+]] = and i1 [[ISARRAY]], [[ISNOTDEL]]
// CK0: br i1 [[CMP1]], label %[[EVALDEL:[^,]+]], label %[[DONE]]
// CK0: [[EVALDEL]]
// CK0-64-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 16
// CK0-32-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 8

// Remove movement mappings and mark as implicit
// CK0-DAG: [[DTYPE:%.+]] = and i64 [[TYPE]], -4
// CK0-DAG: [[DTYPE1:%.+]] = or i64 [[DTYPE]], 512
// CK0: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[BPTR]], ptr [[BEGIN]], i64 [[ARRSIZE]], i64 [[DTYPE1]], {{.*}})
// CK0: br label %[[DONE]]
// CK0: [[DONE]]
// CK0: ret void


// CK0-LABEL: define {{.*}}void @{{.*}}foo{{.*}}
void foo(int a){
  int i = a;
  C c;
  c.a = a;

// CK0-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK0-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK0-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK0-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK0-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK0-DAG: [[MARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 7
// CK0-DAG: store ptr [[MPR:%.+]], ptr [[MARG]]
// CK0-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK0-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK0-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK0-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK0-DAG: [[MPR1:%.+]] = getelementptr inbounds {{.+}}[[MPR]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: store ptr [[VAL:%[^,]+]], ptr [[BP1]]
// CK0-DAG: store ptr [[VAL]], ptr [[P1]]
// CK0-DAG: store ptr [[MPRFUNC]], ptr [[MPR1]]
// CK0: call void [[KERNEL_1:@.+]](ptr [[VAL]])
#pragma omp target map(mapper(id), tofrom \
                       : c)
  {
    ++c.a;
  }

  // CK0: [[BP2GEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[OFFLOAD_BP2:%[^,]+]], i32 0, i32 0
  // CK0: store ptr [[CADDR:%[^,]+]], ptr [[BP2GEP]], align
  // CK0: [[P2GEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[OFFLOAD_P2:%[^,]+]], i32 0, i32 0
  // CK0: store ptr [[CADDR]], ptr [[P2GEP]], align
  // CK0: [[MAPPER2GEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[OFFLOAD_MAPPER2:%[^,]+]], i[[SZ]] 0, i[[SZ]] 0
  // CK0: store ptr [[MPRFUNC]], ptr [[MAPPER2GEP]], align
  // CK0: [[BP2:%.+]] = getelementptr inbounds [1 x ptr], ptr [[OFFLOAD_BP2]], i32 0, i32 0
  // CK0: [[P2:%.+]] = getelementptr inbounds [1 x ptr], ptr [[OFFLOAD_P2]], i32 0, i32 0
  // CK0-32: [[TASK:%.+]] = call ptr @__kmpc_omp_target_task_alloc(ptr {{@.+}}, i32 {{%.+}}, i32 1, i32 40, i32 4, ptr [[TASK_ENTRY:@.+]], i64 -1)
  // CK0-64: [[TASK:%.+]] = call ptr @__kmpc_omp_target_task_alloc(ptr {{@.+}}, i32 {{%.+}}, i32 1, i64 72, i64 8, ptr [[TASK_ENTRY:@.+]], i64 -1)
  // CK0: [[TASK_WITH_PRIVATES:%.+]] = getelementptr inbounds nuw [[KMP_TASK_T_WITH_PRIVATES]], ptr [[TASK]], i32 0, i32 1
  // CK0: {{.+}} = call i32 @__kmpc_omp_task(ptr @1, i32 {{.+}}, ptr [[TASK]])
  #pragma omp target map(mapper(id),tofrom: c) nowait
  {
    ++c.a;
  }

// CK0-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 0, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK0-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK0-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK0-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK0-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK0-DAG: [[MARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 7
// CK0-DAG: store ptr [[MPRGEP:%.+]], ptr [[MARG]]
// CK0-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK0-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK0-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK0-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK0-DAG: [[MPR1:%.+]] = getelementptr inbounds {{.+}}[[MPRGEP]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: store ptr [[VAL:%[^,]+]], ptr [[BP1]]
// CK0-DAG: store ptr [[VAL]], ptr [[P1]]
// CK0-DAG: store ptr [[MPRFUNC]], ptr [[MPR1]]
// CK0: call void [[KERNEL_3:@.+]](ptr [[VAL]])
#pragma omp target teams map(mapper(id), to \
                             : c)
  {
    ++c.a;
  }

  // CK0-32: [[TASK_1:%.+]] = call ptr @__kmpc_omp_target_task_alloc(ptr {{@.+}}, i32 {{%.+}}, i32 1, i32 40, i32 4, ptr [[TASK_ENTRY_1:@.+]], i64 -1)
  // CK0-64: [[TASK_1:%.+]] = call ptr @__kmpc_omp_target_task_alloc(ptr {{@.+}}, i32 {{%.+}}, i32 1, i64 72, i64 8, ptr [[TASK_ENTRY_1:@.+]], i64 -1)
  // CK0: [[TASK_CAST_GET_1:%.+]] = getelementptr inbounds nuw [[KMP_TASK_T_WITH_PRIVATES_1]], ptr [[TASK_1]], i32 0, i32 0
  // CK0: {{.+}} = getelementptr inbounds nuw [[KMP_TASK_T]], ptr [[TASK_CAST_GET_1]], i32 0, i32 0
  // CK0: {{.+}} = call i32 @__kmpc_omp_task(ptr @1, i32 {{.+}}, ptr [[TASK_1]])
  #pragma omp target teams map(mapper(id),to: c) nowait
  {
    ++c.a;
  }

  // CK0-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 {{.+}}, i32 1, ptr [[BPGEP:%[0-9]+]], ptr [[PGEP:%[0-9]+]], {{.+}}[[EDSIZES]], {{.+}}[[EDTYPES]], ptr null, ptr [[MPR:%.+]])
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK0-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK0-DAG: [[MPR1:%.+]] = getelementptr inbounds {{.+}}[[MPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: store ptr [[VAL:%[^,]+]], ptr [[BP1]]
  // CK0-DAG: store ptr [[VAL]], ptr [[P1]]
  // CK0-DAG: store ptr [[MPRFUNC]], ptr [[MPR1]]
  #pragma omp target enter data map(mapper(id),to: c)

  // CK0-DAG: call i32 @__kmpc_omp_task(ptr @{{[^,]+}}, i32 %{{[^,]+}}, ptr [[TASK_2:%.+]])
  // CK0-DAG: [[TASK_2]] = call ptr @__kmpc_omp_target_task_alloc(ptr @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i[[sz]] {{40|72}}, i[[sz]] 1, ptr [[OMP_TASK_ENTRY_18:@[^,]+]], i64 -1)
  // CK0-DAG: [[PRIVATES:%.+]] = getelementptr inbounds nuw [[KMP_TASK_T_WITH_PRIVATES_4]], ptr [[TASK_2]], i32 0, i32 1
  // CK0-32-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_5]], ptr [[PRIVATES]], i32 0, i32 1
  // CK0-64-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_5]], ptr [[PRIVATES]], i32 0, i32 0
  // CK0-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPBPGEP]], ptr align {{4|8}} [[BPGEP:%.+]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds [1 x ptr], ptr [[BP:%.+]], i32 0, i32 0
  // CK0-DAG: [[BPGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP]], i32 0, i32 0
  // CK0-DAG: store ptr [[C:%[^,]+]], ptr [[BPGEP]], align
  // CK0-32-DAG: [[FPPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_5]], ptr [[PRIVATES]], i32 0, i32 2
  // CK0-64-DAG: [[FPPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_5]], ptr [[PRIVATES]], i32 0, i32 1
  // CK0-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPPGEP]], ptr align {{4|8}} [[PGEP:%.+]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[PGEP]] = getelementptr inbounds [1 x ptr], ptr [[P:%.+]], i32 0, i32 0
  // CK0-DAG: [[PGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P]], i32 0, i32 0
  // CK0-DAG: store ptr [[C]], ptr [[PGEP]], align
  // CK0-32-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_5]], ptr [[PRIVATES]], i32 0, i32 0
  // CK0-64-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_5]], ptr [[PRIVATES]], i32 0, i32 2
  // CK0-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPSZGEP]], ptr align {{4|8}} [[EDNWSIZES]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[FPMPRGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_5]], ptr [[PRIVATES]], i32 0, i32 3
  // CK0-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPMPRGEP]], ptr align {{4|8}} [[MPR:%.+]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[MPRGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[MPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: store ptr [[MPRFUNC]], ptr [[MPRGEP]], align
  #pragma omp target enter data map(mapper(id),to: c) nowait

  // CK0-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 {{.+}}, i32 1, ptr [[BPGEP:%[0-9]+]], ptr [[PGEP:%[0-9]+]], {{.+}}[[EXDSIZES]], {{.+}}[[EXDTYPES]], ptr null, ptr [[MPR:%.+]])
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK0-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK0-DAG: [[MPR1:%.+]] = getelementptr inbounds {{.+}}[[MPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: store ptr [[VAL:%[^,]+]], ptr [[BP1]]
  // CK0-DAG: store ptr [[VAL]], ptr [[P1]]
  // CK0-DAG: store ptr [[MPRFUNC]], ptr [[MPR1]]
  #pragma omp target exit data map(mapper(id),from: c)

  // CK0-DAG: call i32 @__kmpc_omp_task(ptr @{{[^,]+}}, i32 %{{[^,]+}}, ptr [[TASK_3:%.+]])
  // CK0-DAG: [[TASK_3]] = call ptr @__kmpc_omp_target_task_alloc(ptr @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i[[sz]] {{40|72}}, i[[sz]] 1, ptr [[OMP_TASK_ENTRY_25:@[^,]+]], i64 -1)
  // CK0-DAG: [[PRIVATES:%.+]] = getelementptr inbounds nuw [[KMP_TASK_T_WITH_PRIVATES_7]], ptr [[TASK_3]], i32 0, i32 1
  // CK0-32-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_8]], ptr [[PRIVATES]], i32 0, i32 1
  // CK0-64-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_8]], ptr [[PRIVATES]], i32 0, i32 0
  // CK0-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPBPGEP]], ptr align {{4|8}} [[BPGEP:%.+]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds [1 x ptr], ptr [[BP:%.+]], i32 0, i32 0
  // CK0-DAG: [[BPGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP]], i32 0, i32 0
  // CK0-DAG: store ptr [[C:%[^,]+]], ptr [[BPGEP]], align
  // CK0-32-DAG: [[FPPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_8]], ptr [[PRIVATES]], i32 0, i32 2
  // CK0-64-DAG: [[FPPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_8]], ptr [[PRIVATES]], i32 0, i32 1
  // CK0-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPPGEP]], ptr align {{4|8}} [[PGEP:%.+]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[PGEP]] = getelementptr inbounds [1 x ptr], ptr [[P:%.+]], i32 0, i32 0
  // CK0-DAG: [[PGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P]], i32 0, i32 0
  // CK0-DAG: store ptr [[C]], ptr [[PGEP]], align
  // CK0-32-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_8]], ptr [[PRIVATES]], i32 0, i32 0
  // CK0-64-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_8]], ptr [[PRIVATES]], i32 0, i32 2
  // CK0-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPSZGEP]], ptr align {{4|8}} [[EXDNWSIZES]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[FPMPRGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_8]], ptr [[PRIVATES]], i32 0, i32 3
  // CK0-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPMPRGEP]], ptr align {{4|8}} [[MPR:%.+]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[MPRGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[MPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: store ptr [[MPRFUNC]], ptr [[MPRGEP]], align
  #pragma omp target exit data map(mapper(id),from: c) nowait

  // CK0-DAG: call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[TGEPBP:%.+]], ptr [[TGEPP:%.+]], ptr [[TSIZES]], ptr [[TTYPES]], ptr null, ptr [[TMPR:%.+]])
  // CK0-DAG: [[TGEPBP]] = getelementptr inbounds {{.+}}[[TBP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[TGEPP]] = getelementptr inbounds {{.+}}[[TP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[TBP0:%.+]] = getelementptr inbounds {{.+}}[[TBP]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[TP0:%.+]] = getelementptr inbounds {{.+}}[[TP]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[TMPR1:%.+]] = getelementptr inbounds {{.+}}[[TMPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: store ptr [[VAL]], ptr [[TBP0]]
  // CK0-DAG: store ptr [[VAL]], ptr [[TP0]]
  // CK0-DAG: store ptr [[MPRFUNC]], ptr [[TMPR1]]
  #pragma omp target update to(mapper(id): c)

  // CK0-DAG: call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[FGEPBP:%.+]], ptr [[FGEPP:%.+]], ptr [[FSIZES]], ptr [[FTYPES]], ptr null, ptr [[FMPR:%.+]])
  // CK0-DAG: [[FGEPBP]] = getelementptr inbounds {{.+}}[[FBP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[FGEPP]] = getelementptr inbounds {{.+}}[[FP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[FBP0:%.+]] = getelementptr inbounds {{.+}}[[FBP]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[FP0:%.+]] = getelementptr inbounds {{.+}}[[FP]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[FMPR1:%.+]] = getelementptr inbounds {{.+}}[[FMPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: store ptr [[VAL]], ptr [[FBP0]]
  // CK0-DAG: store ptr [[VAL]], ptr [[FP0]]
  // CK0-DAG: store ptr [[MPRFUNC]], ptr [[FMPR1]]
  #pragma omp target update from(mapper(id): c)

  // CK0-DAG: call i32 @__kmpc_omp_task(ptr @{{[^,]+}}, i32 %{{[^,]+}}, ptr [[TASK_4:%.+]])
  // CK0-DAG: [[TASK_4]] = call ptr @__kmpc_omp_target_task_alloc(ptr @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i[[sz]] {{40|72}}, i[[sz]] 1, ptr [[OMP_TASK_ENTRY_34:@[^,]+]], i64 -1)
  // CK0-DAG: [[PRIVATES:%.+]] = getelementptr inbounds nuw [[KMP_TASK_T_WITH_PRIVATES_10]], ptr [[TASK_4]], i32 0, i32 1
  // CK0-32-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_11]], ptr [[PRIVATES]], i32 0, i32 1
  // CK0-64-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_11]], ptr [[PRIVATES]], i32 0, i32 0
  // CK0-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPBPGEP]], ptr align {{4|8}} [[BPGEP:%.+]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds [1 x ptr], ptr [[BP:%.+]], i32 0, i32 0
  // CK0-DAG: [[BPGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP]], i32 0, i32 0
  // CK0-DAG: store ptr [[C:%[^,]+]], ptr [[BPGEP]], align
  // CK0-32-DAG: [[FPPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_11]], ptr [[PRIVATES]], i32 0, i32 2
  // CK0-64-DAG: [[FPPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_11]], ptr [[PRIVATES]], i32 0, i32 1
  // CK0-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPPGEP]], ptr align {{4|8}} [[PGEP:%.+]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[PGEP]] = getelementptr inbounds [1 x ptr], ptr [[P:%.+]], i32 0, i32 0
  // CK0-DAG: [[PGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P]], i32 0, i32 0
  // CK0-DAG: store ptr [[C]], ptr [[PGEP]], align
  // CK0-32-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_11]], ptr [[PRIVATES]], i32 0, i32 0
  // CK0-64-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_11]], ptr [[PRIVATES]], i32 0, i32 2
  // CK0-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPSZGEP]], ptr align {{4|8}} [[FNWSIZES]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[FPMPRGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T_11]], ptr [[PRIVATES]], i32 0, i32 3
  // CK0-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPMPRGEP]], ptr align {{4|8}} [[MPR:%.+]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[MPRGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[MPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: store ptr [[MPRFUNC]], ptr [[MPRGEP]], align
  #pragma omp target update from(mapper(id): c) nowait
}


// CK0: define internal void [[KERNEL_1]](ptr {{.+}}[[ARG:%.+]])
// CK0: [[ADDR:%.+]] = alloca ptr,
// CK0: store ptr [[ARG]], ptr [[ADDR]]
// CK0: [[CADDR:%.+]] = load ptr, ptr [[ADDR]]
// CK0: [[CAADDR:%.+]] = getelementptr inbounds nuw %class.C, ptr [[CADDR]], i32 0, i32 0
// CK0: [[VAL:%[^,]+]] = load i32, ptr [[CAADDR]]
// CK0: {{.+}} = add nsw i32 [[VAL]], 1
// CK0: }

// CK0: define internal void [[KERNEL_2:@.+]](ptr {{.+}}[[ARG:%.+]])
// CK0: [[ADDR:%.+]] = alloca ptr,
// CK0: store ptr [[ARG]], ptr [[ADDR]]
// CK0: [[CADDR:%.+]] = load ptr, ptr [[ADDR]]
// CK0: [[CAADDR:%.+]] = getelementptr inbounds nuw %class.C, ptr [[CADDR]], i32 0, i32 0
// CK0: [[VAL:%[^,]+]] = load i32, ptr [[CAADDR]]
// CK0: {{.+}} = add nsw i32 [[VAL]], 1
// CK0: }

// CK0: define internal void [[OUTLINED:@.+]](i32 {{.*}}{{[^,]+}}, ptr noalias noundef [[CTXARG:%.+]])
// CK0-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK0-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK0-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK0-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK0-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK0-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK0-DAG: store ptr [[SIZEGEP:%.+]], ptr [[SARG]]
// CK0-DAG: [[MARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 7
// CK0-DAG: store ptr [[MPRGEP:%.+]], ptr [[MARG]]
// CK0-DAG: [[BPGEP]] = getelementptr inbounds [1 x ptr], ptr [[BPFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[PGEP]] = getelementptr inbounds [1 x ptr], ptr [[PFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[SIZEGEP]] = getelementptr inbounds [1 x i64], ptr [[SIZEFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[MPRGEP]] = getelementptr inbounds [1 x ptr], ptr [[MPRFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[BPFPADDR]] = load ptr, ptr [[FPPTRADDR_BP:%.+]], align
// CK0-DAG: [[PFPADDR]] = load ptr, ptr [[FPPTRADDR_P:%.+]], align
// CK0-DAG: [[SIZEFPADDR]] = load ptr, ptr [[FPPTRADDR_SIZE:%.+]], align
// CK0-DAG: [[MPRFPADDR]] = load ptr, ptr [[FPPTRADDR_MPR:%.+]], align
// CK0-DAG: call void %1(ptr %2, {{.+}}[[FPPTRADDR_BP]], {{.+}}[[FPPTRADDR_P]], {{.+}}[[FPPTRADDR_SIZE]], {{.+}}[[FPPTRADDR_MPR]])
// CK0-DAG: call void [[KERNEL_2:@.+]](ptr [[KERNELARG:%.+]])
// CK0-DAG: [[KERNELARG]] = load ptr, ptr [[KERNELARGGEP:%.+]], align
// CK0-DAG: [[KERNELARGGEP]] = getelementptr inbounds nuw [[ANON_T]], ptr [[CTX:%.+]], i32 0, i32 0
// CK0-DAG: [[CTX]] = load ptr, ptr [[CTXADDR:%.+]], align
// CK0-DAG: store ptr [[CTXARG]], ptr [[CTXADDR]], align
// CK0: }

// CK0: define internal {{.*}}i32 [[TASK_ENTRY]](i32 {{.*}}%0, ptr noalias noundef %1)
// CK0: store ptr %1, ptr [[ADDR:%.+]], align
// CK0: [[TASK_T_WITH_PRIVATES:%.+]] = load ptr, ptr [[ADDR]], align
// CK0: [[TASKGEP:%.+]] = getelementptr inbounds nuw [[KMP_TASK_T_WITH_PRIVATES]], ptr [[TASK_T_WITH_PRIVATES]], i32 0, i32 0
// CK0: [[SHAREDSGEP:%.+]] = getelementptr inbounds nuw [[KMP_TASK_T]], ptr [[TASKGEP]], i32 0, i32 0
// CK0: [[SHAREDS:%.+]] = load ptr, ptr [[SHAREDSGEP]], align
// CK0: [[PRIVATESGEP:%.+]] = getelementptr inbounds nuw [[KMP_TASK_T_WITH_PRIVATES]], ptr [[TASK_T_WITH_PRIVATES]], i32 0, i32 1
// CK0: call void [[OUTLINED]](i32 {{%.+}}, ptr {{%.+}}, ptr [[PRIVATESGEP]], {{.+}}, ptr [[TASK_T_WITH_PRIVATES]], ptr [[SHAREDS]])
// CK0: }

// CK0: define internal void [[OUTLINE_1:@.+]](i32 {{.*}}%.global_tid.{{.+}}, ptr noalias noundef [[CTXARG:%.+]])
// CK0-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 0, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK0-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK0-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK0-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK0-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK0-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK0-DAG: store ptr [[SIZEGEP:%.+]], ptr [[SARG]]
// CK0-DAG: [[MARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 7
// CK0-DAG: store ptr [[MPRGEP:%.+]], ptr [[MARG]]
// CK0-DAG: [[BPGEP]] = getelementptr inbounds [1 x ptr], ptr [[BPFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[PGEP]] = getelementptr inbounds [1 x ptr], ptr [[PFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[SIZEGEP]] = getelementptr inbounds [1 x i64], ptr [[SIZEFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[MPRGEP]] = getelementptr inbounds [1 x ptr], ptr [[MPRFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[BPFPADDR]] = load ptr, ptr [[FPPTRADDR_BP:%.+]], align
// CK0-DAG: [[PFPADDR]] = load ptr, ptr [[FPPTRADDR_P:%.+]], align
// CK0-DAG: [[SIZEFPADDR]] = load ptr, ptr [[FPPTRADDR_SIZE:%.+]], align
// CK0-DAG: [[MPRFPADDR]] = load ptr, ptr [[FPPTRADDR_MPR:%.+]], align
// CK0-DAG: call void %1(ptr %2, {{.+}}[[FPPTRADDR_BP]], {{.+}}[[FPPTRADDR_P]], {{.+}}[[FPPTRADDR_SIZE]], {{.+}}[[FPPTRADDR_MPR]])
// CK0-DAG: call void [[KERNEL_2:@.+]](ptr [[KERNELARG:%.+]])
// CK0-DAG: [[KERNELARG]] = load ptr, ptr [[KERNELARGGEP:%.+]], align
// CK0-DAG: [[KERNELARGGEP]] = getelementptr inbounds nuw [[ANON_T_0]], ptr [[CTX:%.+]], i32 0, i32 0
// CK0-DAG: [[CTX]] = load ptr, ptr [[CTXADDR:%.+]], align
// CK0-DAG: store ptr [[CTXARG]], ptr [[CTXADDR]], align
// CK0: }

// CK0: define internal {{.*}}i32 [[TASK_ENTRY_1]](i32 {{.*}}%0, ptr noalias noundef %1)
// CK0: store ptr %1, ptr [[ADDR:%.+]], align
// CK0: [[TASK_T_WITH_PRIVATES:%.+]] = load ptr, ptr [[ADDR]], align
// CK0: [[TASKGEP:%.+]] = getelementptr inbounds nuw [[KMP_TASK_T_WITH_PRIVATES_1]], ptr [[TASK_T_WITH_PRIVATES]], i32 0, i32 0
// CK0: [[SHAREDSGEP:%.+]] = getelementptr inbounds nuw [[KMP_TASK_T]], ptr [[TASKGEP]], i32 0, i32 0
// CK0: [[SHAREDS:%.+]] = load ptr, ptr [[SHAREDSGEP]], align
// CK0: [[PRIVATESGEP:%.+]] = getelementptr inbounds nuw [[KMP_TASK_T_WITH_PRIVATES_1]], ptr [[TASK_T_WITH_PRIVATES]], i32 0, i32 1
// CK0: call void [[OUTLINE_1]](i32 {{%.+}}, ptr {{%.+}}, ptr [[PRIVATESGEP]], {{.+}}, ptr [[TASK_T_WITH_PRIVATES]], ptr [[SHAREDS]])
// CK0: }

// CK0: define internal void [[OMP_OUTLINED_16:@.+]](i32{{.*}} %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}
// CK0-DAG: call void @__tgt_target_data_begin_nowait_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[BP:%[^,]+]], ptr [[P:%[^,]+]], ptr [[SZ:%[^,]+]], ptr [[EDNWTYPES]], ptr null, ptr [[MPR:%.+]], i32 0, ptr null, i32 0, ptr null)
// CK0-DAG: [[BP]] = getelementptr inbounds [1 x ptr], ptr [[BPADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[P]] = getelementptr inbounds [1 x ptr], ptr [[PADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[SZ]] = getelementptr inbounds [1 x i64], ptr [[SZADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[MPR]] = getelementptr inbounds [1 x ptr], ptr [[MPRADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[BPADDR]] = load ptr, ptr [[FPBPADDR:%[^,]+]], align
// CK0-DAG: [[PADDR]] = load ptr, ptr [[FPPADDR:%[^,]+]], align
// CK0-DAG: [[SZADDR]] = load ptr, ptr [[FPSZADDR:%[^,]+]], align
// CK0-DAG: [[MPRADDR]] = load ptr, ptr [[FPMPRADDR:%[^,]+]], align
// CK0-DAG: call void %{{.+}}(ptr %{{[^,]+}}, ptr [[FPBPADDR]], ptr [[FPPADDR]], ptr [[FPSZADDR]], ptr [[FPMPRADDR]])
// CK0: ret void
// CK0: }

// CK0: define internal {{.*}}i32 [[OMP_TASK_ENTRY_18]](i32 {{.*}}%{{[^,]+}}, ptr noalias noundef %{{[^,]+}})
// CK0:   call void [[OMP_OUTLINED_16]]
// CK0:   ret i32 0
// CK0: }

// CK0: define internal void [[OMP_OUTLINED_23:@.+]](i32{{.*}} %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}
// CK0-DAG: call void @__tgt_target_data_end_nowait_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[BP:%[^,]+]], ptr [[P:%[^,]+]], ptr [[SZ:%[^,]+]], ptr [[EXDNWTYPES]], ptr null, ptr [[MPR:%.+]], i32 0, ptr null, i32 0, ptr null)
// CK0-DAG: [[BP]] = getelementptr inbounds [1 x ptr], ptr [[BPADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[P]] = getelementptr inbounds [1 x ptr], ptr [[PADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[SZ]] = getelementptr inbounds [1 x i64], ptr [[SZADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[MPR]] = getelementptr inbounds [1 x ptr], ptr [[MPRADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[BPADDR]] = load ptr, ptr [[FPBPADDR:%[^,]+]], align
// CK0-DAG: [[PADDR]] = load ptr, ptr [[FPPADDR:%[^,]+]], align
// CK0-DAG: [[SZADDR]] = load ptr, ptr [[FPSZADDR:%[^,]+]], align
// CK0-DAG: [[MPRADDR]] = load ptr, ptr [[FPMPRADDR:%[^,]+]], align
// CK0-DAG: call void %{{.+}}(ptr %{{[^,]+}}, ptr [[FPBPADDR]], ptr [[FPPADDR]], ptr [[FPSZADDR]], ptr [[FPMPRADDR]])
// CK0: }

// CK0: define internal {{.*}}i32 [[OMP_TASK_ENTRY_25]](i32 {{.*}}%{{[^,]+}}, ptr noalias noundef %{{[^,]+}})
// CK0:   call void [[OMP_OUTLINED_23]]
// CK0:   ret i32 0
// CK0: }

// CK0: define internal void [[OMP_OUTLINED_32:@.+]](i32{{.*}} %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}, ptr noalias noundef %{{[^,]+}}
// CK0-DAG: call void @__tgt_target_data_update_nowait_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[BP:%[^,]+]], ptr [[P:%[^,]+]], ptr [[SZ:%[^,]+]], ptr [[FNWTYPES]], ptr null, ptr [[MPR:%.+]], i32 0, ptr null, i32 0, ptr null)
// CK0-DAG: [[BP]] = getelementptr inbounds [1 x ptr], ptr [[BPADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[P]] = getelementptr inbounds [1 x ptr], ptr [[PADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[SZ]] = getelementptr inbounds [1 x i64], ptr [[SZADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[MPR]] = getelementptr inbounds [1 x ptr], ptr [[MPRADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[BPADDR]] = load ptr, ptr [[FPBPADDR:%[^,]+]], align
// CK0-DAG: [[PADDR]] = load ptr, ptr [[FPPADDR:%[^,]+]], align
// CK0-DAG: [[SZADDR]] = load ptr, ptr [[FPSZADDR:%[^,]+]], align
// CK0-DAG: [[MPRADDR]] = load ptr, ptr [[FPMPRADDR:%[^,]+]], align
// CK0-DAG: call void %{{.+}}(ptr %{{[^,]+}}, ptr [[FPBPADDR]], ptr [[FPPADDR]], ptr [[FPSZADDR]], ptr [[FPMPRADDR]])
// CK0: }

// CK0: define internal {{.*}}i32 [[OMP_TASK_ENTRY_34]](i32 {{.*}}%{{[^,]+}}, ptr noalias noundef %{{[^,]+}})
// CK0:   call void [[OMP_OUTLINED_32]]
// CK0:   ret i32 0
// CK0: }

#endif // CK0


///==========================================================================///
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix CK1 --check-prefix CK1-64 %s
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix CK1 --check-prefix CK1-64 %s
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix CK1 --check-prefix CK1-32 %s
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix CK1 --check-prefix CK1-32 %s

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s

#ifdef CK1
// C++ template

template <class T>
class C {
public:
  T a;
};

#pragma omp declare mapper(id: C<int> s) map(s.a)

// CK1-LABEL: define {{.*}}void @.omp_mapper.{{.*}}C{{.*}}.id{{.*}}(ptr{{.*}}, ptr{{.*}}, ptr{{.*}}, i64{{.*}}, i64{{.*}}, ptr{{.*}})
// CK1: store ptr %{{[^,]+}}, ptr [[HANDLEADDR:%[^,]+]]
// CK1: store ptr %{{[^,]+}}, ptr [[BPTRADDR:%[^,]+]]
// CK1: store ptr %{{[^,]+}}, ptr [[VPTRADDR:%[^,]+]]
// CK1: store i64 %{{[^,]+}}, ptr [[SIZEADDR:%[^,]+]]
// CK1: store i64 %{{[^,]+}}, ptr [[TYPEADDR:%[^,]+]]
// CK1-DAG: [[BYTESIZE:%.+]] = load i64, ptr [[SIZEADDR]]
// CK1-DAG: [[SIZE:%.+]] = udiv exact i64 [[BYTESIZE]], 4
// CK1-DAG: [[TYPE:%.+]] = load i64, ptr [[TYPEADDR]]
// CK1-DAG: [[HANDLE:%.+]] = load ptr, ptr [[HANDLEADDR]]
// CK1-DAG: [[BPTR:%.+]] = load ptr, ptr [[BPTRADDR]]
// CK1-DAG: [[BEGIN:%.+]] = load ptr, ptr [[VPTRADDR]]
// CK1-DAG: [[PTREND:%.+]] = getelementptr %class.C, ptr [[BEGIN]], i64 [[SIZE]]
// CK1-DAG: [[ISARRAY:%.+]] = icmp sgt i64 [[SIZE]], 1
// CK1-DAG: [[PTRSNE:%.+]] = icmp ne ptr [[BPTR]], [[BEGIN]]
// CK1-DAG: [[PTRANDOBJ:%.+]] = and i64 [[TYPE]], 16
// CK1-DAG: [[ISPTRANDOBJ:%.+]] = icmp ne i64 [[PTRANDOBJ]], 0
// CK1-DAG: [[CMPA:%.+]] = and i1 [[PTRSNE]], [[ISPTRANDOBJ]]
// CK1-DAG: [[CMP:%.+]] = or i1 [[ISARRAY]], [[CMPA]]
// CK1-DAG: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK1-DAG: [[ISNOTDEL:%.+]] = icmp eq i64 [[TYPEDEL]], 0
// CK1-DAG: [[CMP1:%.+]] = and i1 [[CMP]], [[ISNOTDEL]]
// CK1: br i1 [[CMP1]], label %[[INITEVALDEL:[^,]+]], label %[[LHEAD:[^,]+]]

// CK1: [[INITEVALDEL]]
// CK1-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 4

// Remove movement mappings and mark as implicit
// CK1-DAG: [[ITYPE:%.+]] = and i64 [[TYPE]], -4
// CK1-DAG: [[ITYPE1:%.+]] = or i64 [[ITYPE]], 512
// CK1: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[BPTR]], ptr [[BEGIN]], i64 [[ARRSIZE]], i64 [[ITYPE1]], {{.*}})
// CK1: br label %[[LHEAD:[^,]+]]

// CK1: [[LHEAD]]
// CK1: [[ISEMPTY:%.+]] = icmp eq ptr [[BEGIN]], [[PTREND]]
// CK1: br i1 [[ISEMPTY]], label %[[DONE:[^,]+]], label %[[LBODY:[^,]+]]
// CK1: [[LBODY]]
// CK1: [[PTR:%.+]] = phi ptr [ [[BEGIN]], %{{.+}} ], [ [[PTRNEXT:%.+]], %[[LCORRECT:[^,]+]] ]
// CK1-DAG: [[ABEGIN:%.+]] = getelementptr inbounds nuw %class.C, ptr [[PTR]], i32 0, i32 0
// CK1-DAG: [[PRESIZE:%.+]] = call i64 @__tgt_mapper_num_components(ptr [[HANDLE]])
// CK1-DAG: [[SHIPRESIZE:%.+]] = shl i64 [[PRESIZE]], 48
// CK1-DAG: [[MEMBERTYPE:%.+]] = add nuw i64 3, [[SHIPRESIZE]]
// CK1-DAG: [[TYPETF:%.+]] = and i64 [[TYPE]], 3
// CK1-DAG: [[ISALLOC:%.+]] = icmp eq i64 [[TYPETF]], 0
// CK1-DAG: br i1 [[ISALLOC]], label %[[ALLOC:[^,]+]], label %[[ALLOCELSE:[^,]+]]
// CK1-DAG: [[ALLOC]]
// CK1-DAG: [[ALLOCTYPE:%.+]] = and i64 [[MEMBERTYPE]], -4
// CK1-DAG: br label %[[TYEND:[^,]+]]
// CK1-DAG: [[ALLOCELSE]]
// CK1-DAG: [[ISTO:%.+]] = icmp eq i64 [[TYPETF]], 1
// CK1-DAG: br i1 [[ISTO]], label %[[TO:[^,]+]], label %[[TOELSE:[^,]+]]
// CK1-DAG: [[TO]]
// CK1-DAG: [[TOTYPE:%.+]] = and i64 [[MEMBERTYPE]], -3
// CK1-DAG: br label %[[TYEND]]
// CK1-DAG: [[TOELSE]]
// CK1-DAG: [[ISFROM:%.+]] = icmp eq i64 [[TYPETF]], 2
// CK1-DAG: br i1 [[ISFROM]], label %[[FROM:[^,]+]], label %[[TYEND]]
// CK1-DAG: [[FROM]]
// CK1-DAG: [[FROMTYPE:%.+]] = and i64 [[MEMBERTYPE]], -2
// CK1-DAG: br label %[[TYEND]]
// CK1-DAG: [[TYEND]]
// CK1-DAG: [[TYPE1:%.+]] = phi i64 [ [[ALLOCTYPE]], %[[ALLOC]] ], [ [[TOTYPE]], %[[TO]] ], [ [[FROMTYPE]], %[[FROM]] ], [ [[MEMBERTYPE]], %[[TOELSE]] ]
// CK1: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[PTR]], ptr [[ABEGIN]], i64 4, i64 [[TYPE1]], {{.*}})
// CK1: [[PTRNEXT]] = getelementptr %class.C, ptr [[PTR]], i32 1
// CK1: [[ISDONE:%.+]] = icmp eq ptr [[PTRNEXT]], [[PTREND]]
// CK1: br i1 [[ISDONE]], label %[[LEXIT:[^,]+]], label %[[LBODY]]

// CK1: [[LEXIT]]
// CK1: [[ISARRAY:%.+]] = icmp sgt i64 [[SIZE]], 1
// CK1: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK1: [[ISNOTDEL:%.+]] = icmp ne i64 [[TYPEDEL]], 0
// CK1: [[CMP1:%.+]] = and i1 [[ISARRAY]], [[ISNOTDEL]]
// CK1-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 4

// Remove movement mappings and mark as implicit
// CK1-DAG: [[DTYPE:%.+]] = and i64 [[TYPE]], -4
// CK1-DAG: [[DTYPE1:%.+]] = or i64 [[DTYPE]], 512
// CK1: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[BPTR]], ptr [[BEGIN]], i64 [[ARRSIZE]], i64 [[DTYPE1]], {{.*}})
// CK1: br label %[[DONE]]
// CK1: [[DONE]]
// CK1: ret void

#endif // CK1


///==========================================================================///
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix CK2 --check-prefix CK2-64 %s
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix CK2 --check-prefix CK2-64 %s
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix CK2 --check-prefix CK2-32 %s
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix CK2 --check-prefix CK2-32 %s

// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s

#ifdef CK2
// Nested mappers.

class B {
public:
  double a;
};

class C {
public:
  double a;
  B b;
};

#pragma omp declare mapper(B s) map(s.a)

#pragma omp declare mapper(id: C s) map(s.b)

// CK2: define {{.*}}void [[BMPRFUNC:@[.]omp_mapper[.].*B[.]default]](ptr{{.*}}, ptr{{.*}}, ptr{{.*}}, i64{{.*}}, i64{{.*}}, ptr{{.*}})

// CK2-LABEL: define {{.*}}void @.omp_mapper.{{.*}}C{{.*}}.id(ptr{{.*}}, ptr{{.*}}, ptr{{.*}}, i64{{.*}}, i64{{.*}}, ptr{{.*}})
// CK2: store ptr %{{[^,]+}}, ptr [[HANDLEADDR:%[^,]+]]
// CK2: store ptr %{{[^,]+}}, ptr [[BPTRADDR:%[^,]+]]
// CK2: store ptr %{{[^,]+}}, ptr [[VPTRADDR:%[^,]+]]
// CK2: store i64 %{{[^,]+}}, ptr [[SIZEADDR:%[^,]+]]
// CK2: store i64 %{{[^,]+}}, ptr [[TYPEADDR:%[^,]+]]
// CK2-DAG: [[BYTESIZE:%.+]] = load i64, ptr [[SIZEADDR]]
// CK2-DAG: [[SIZE:%.+]] = udiv exact i64 [[BYTESIZE]], 16
// CK2-DAG: [[TYPE:%.+]] = load i64, ptr [[TYPEADDR]]
// CK2-DAG: [[HANDLE:%.+]] = load ptr, ptr [[HANDLEADDR]]
// CK2-DAG: [[BPTR:%.+]] = load ptr, ptr [[BPTRADDR]]
// CK2-DAG: [[BEGIN:%.+]] = load ptr, ptr [[VPTRADDR]]
// CK2-DAG: [[PTREND:%.+]] = getelementptr %class.C, ptr [[BEGIN]], i64 [[SIZE]]
// CK2-DAG: [[ISARRAY:%.+]] = icmp sgt i64 [[SIZE]], 1
// CK2-DAG: [[PTRSNE:%.+]] = icmp ne ptr [[BPTR]], [[BEGIN]]
// CK2-DAG: [[PTRANDOBJ:%.+]] = and i64 [[TYPE]], 16
// CK2-DAG: [[ISPTRANDOBJ:%.+]] = icmp ne i64 [[PTRANDOBJ]], 0
// CK2-DAG: [[CMPA:%.+]] = and i1 [[PTRSNE]], [[ISPTRANDOBJ]]
// CK2-DAG: [[CMP:%.+]] = or i1 [[ISARRAY]], [[CMPA]]
// CK2-DAG: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK2-DAG: [[ISNOTDEL:%.+]] = icmp eq i64 [[TYPEDEL]], 0
// CK2-DAG: [[CMP1:%.+]] = and i1 [[CMP]], [[ISNOTDEL]]
// CK2: br i1 [[CMP1]], label %[[INITEVALDEL:[^,]+]], label %[[LHEAD:[^,]+]]

// CK2: [[INITEVALDEL]]
// CK2-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 16

// Remove movement mappings and mark as implicit
// CK2-DAG: [[ITYPE:%.+]] = and i64 [[TYPE]], -4
// CK2-DAG: [[ITYPE1:%.+]] = or i64 [[ITYPE]], 512
// CK2: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[BPTR]], ptr [[BEGIN]], i64 [[ARRSIZE]], i64 [[ITYPE1]], {{.*}})
// CK2: br label %[[LHEAD:[^,]+]]

// CK2: [[LHEAD]]
// CK2: [[ISEMPTY:%.+]] = icmp eq ptr [[BEGIN]], [[PTREND]]
// CK2: br i1 [[ISEMPTY]], label %[[DONE:[^,]+]], label %[[LBODY:[^,]+]]
// CK2: [[LBODY]]
// CK2: [[PTR:%.+]] = phi ptr [ [[BEGIN]], %{{.+}} ], [ [[PTRNEXT:%.+]], %[[LCORRECT:[^,]+]] ]
// CK2-DAG: [[BBEGIN:%.+]] = getelementptr inbounds nuw %class.C, ptr [[PTR]], i32 0, i32 1
// CK2-DAG: [[PRESIZE:%.+]] = call i64 @__tgt_mapper_num_components(ptr [[HANDLE]])
// CK2-DAG: [[SHIPRESIZE:%.+]] = shl i64 [[PRESIZE]], 48
// CK2-DAG: [[MEMBERTYPE:%.+]] = add nuw i64 3, [[SHIPRESIZE]]
// CK2-DAG: [[TYPETF:%.+]] = and i64 [[TYPE]], 3
// CK2-DAG: [[ISALLOC:%.+]] = icmp eq i64 [[TYPETF]], 0
// CK2-DAG: br i1 [[ISALLOC]], label %[[ALLOC:[^,]+]], label %[[ALLOCELSE:[^,]+]]
// CK2-DAG: [[ALLOC]]
// CK2-DAG: [[ALLOCTYPE:%.+]] = and i64 [[MEMBERTYPE]], -4
// CK2-DAG: br label %[[TYEND:[^,]+]]
// CK2-DAG: [[ALLOCELSE]]
// CK2-DAG: [[ISTO:%.+]] = icmp eq i64 [[TYPETF]], 1
// CK2-DAG: br i1 [[ISTO]], label %[[TO:[^,]+]], label %[[TOELSE:[^,]+]]
// CK2-DAG: [[TO]]
// CK2-DAG: [[TOTYPE:%.+]] = and i64 [[MEMBERTYPE]], -3
// CK2-DAG: br label %[[TYEND]]
// CK2-DAG: [[TOELSE]]
// CK2-DAG: [[ISFROM:%.+]] = icmp eq i64 [[TYPETF]], 2
// CK2-DAG: br i1 [[ISFROM]], label %[[FROM:[^,]+]], label %[[TYEND]]
// CK2-DAG: [[FROM]]
// CK2-DAG: [[FROMTYPE:%.+]] = and i64 [[MEMBERTYPE]], -2
// CK2-DAG: br label %[[TYEND]]
// CK2-DAG: [[TYEND]]
// CK2-DAG: [[TYPE1:%.+]] = phi i64 [ [[ALLOCTYPE]], %[[ALLOC]] ], [ [[TOTYPE]], %[[TO]] ], [ [[FROMTYPE]], %[[FROM]] ], [ [[MEMBERTYPE]], %[[TOELSE]] ]
// CK2: call void [[BMPRFUNC]](ptr [[HANDLE]], ptr [[PTR]], ptr [[BBEGIN]], i64 8, i64 [[TYPE1]], {{.*}})
// CK2: [[PTRNEXT]] = getelementptr %class.C, ptr [[PTR]], i32 1
// CK2: [[ISDONE:%.+]] = icmp eq ptr [[PTRNEXT]], [[PTREND]]
// CK2: br i1 [[ISDONE]], label %[[LEXIT:[^,]+]], label %[[LBODY]]

// CK2: [[LEXIT]]
// CK2: [[ISARRAY:%.+]] = icmp sgt i64 [[SIZE]], 1
// CK2: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK2: [[ISNOTDEL:%.+]] = icmp ne i64 [[TYPEDEL]], 0
// CK2: [[CMP1:%.+]] = and i1 [[ISARRAY]], [[ISNOTDEL]]
// CK2: br i1 [[CMP1]], label %[[EVALDEL:[^,]+]], label %[[DONE]]
// CK2: [[EVALDEL]]
// CK2-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 16

// Remove movement mappings and mark as implicit
// CK2-DAG: [[DTYPE:%.+]] = and i64 [[TYPE]], -4
// CK2-DAG: [[DTYPE1:%.+]] = or i64 [[DTYPE]], 512
// CK2: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[BPTR]], ptr [[BEGIN]], i64 [[ARRSIZE]], i64 [[DTYPE1]], {{.*}})
// CK2: br label %[[DONE]]
// CK2: [[DONE]]
// CK2: ret void

#endif // CK2


///==========================================================================///
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix CK3 %s
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix CK3 %s
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix CK3 %s
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix CK3 %s

// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s

#ifdef CK3
// map of array sections and nested components.

// CK3-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}.region_id = weak constant i8 0
// CK3-DAG: [[SIZES:@.+]] = {{.+}}constant [2 x i64] [i64 {{8|16}}, i64 {{80|160}}]
// CK3-DAG: [[TYPES:@.+]] = {{.+}}constant [2 x i64] [i64 35, i64 35]

class C {
public:
  int a;
  double *b;
};

class B {
public:
  C c;
};

#pragma omp declare mapper(id: C s) map(s.a, s.b[0:2])

// CK3: define {{.*}}void [[MPRFUNC:@[.]omp_mapper[.].*C[.]id]](ptr{{.*}}, ptr{{.*}}, ptr{{.*}}, i64{{.*}}, i64{{.*}}, ptr{{.*}})

// CK3-LABEL: define {{.*}}void @{{.*}}foo{{.*}}
void foo(int a){
  // CK3-DAG: [[CVAL:%.+]] = alloca [10 x %class.C]
  // CK3-DAG: [[BVAL:%.+]] = alloca %class.B
  C c[10];
  B b;

  // CK3-DAG: [[BC:%.+]] = getelementptr inbounds nuw %class.B, ptr [[BVAL]], i32 0, i32 0

// CK3-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK3-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK3-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK3-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK3-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK3-DAG: [[MARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 7
// CK3-DAG: store ptr [[MPR:%.+]], ptr [[MARG]]
// CK3-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK3-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK3-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK3-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK3-DAG: [[MPR1:%.+]] = getelementptr inbounds {{.+}}[[MPR]], i{{64|32}} 0, i{{64|32}} 0
// CK3-DAG: store ptr [[BVAL]], ptr [[BP1]]
// CK3-DAG: store ptr [[BC]], ptr [[P1]]
// CK3-DAG: store ptr [[MPRFUNC]], ptr [[MPR1]]
// CK3-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
// CK3-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
// CK3-DAG: [[MPR2:%.+]] = getelementptr inbounds {{.+}}[[MPR]], i{{64|32}} 0, i{{64|32}} 1
// CK3-DAG: store ptr [[CVAL]], ptr [[BP2]]
// CK3-DAG: [[CVALGEP:%.+]] = getelementptr inbounds {{.+}}[[CVAL]], i{{64|32}} 0, i{{64|32}} 0
// CK3-DAG: store ptr [[CVALGEP]], ptr [[P2]]
// CK3-DAG: store ptr [[MPRFUNC]], ptr [[MPR2]]
// CK3: call void [[KERNEL:@.+]](ptr [[BVAL]], ptr [[CVAL]])
#pragma omp target map(mapper(id), tofrom \
                       : c [0:10], b.c)
  for (int i = 0; i < 10; i++) {
    b.c.a += ++c[i].a;
  }
}


// CK3: define internal void [[KERNEL]](ptr {{[^,]+}}, ptr {{[^,]+}})

#endif // CK3

///==========================================================================///
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix CK4 --check-prefix CK4-64 %s
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix CK4 --check-prefix CK4-64 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix CK4 --check-prefix CK4-32 %s
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix CK4 --check-prefix CK4-32 %s

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s

#ifdef CK4
// Mapper function code generation and runtime interface.

// CK4-64: [[TSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK4-32: [[TSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// PRESENT=0x1000 | TO=0x1 = 0x1001
// CK4: [[TTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1001]]]

// CK4-64: [[FSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK4-32: [[FSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// PRESENT=0x1000 | FROM=0x2 = 0x1002
// CK4: [[FTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1002]]]

class C {
public:
  int a;
  double *b;
};

#pragma omp declare mapper(id: C s) map(s.a, s.b[0:2])

// CK4: define {{.*}}void [[MPRFUNC:@[.]omp_mapper[.].*C[.]id]](ptr{{.*}}, ptr{{.*}}, ptr{{.*}}, i64{{.*}}, i64{{.*}}, ptr{{.*}})
// CK4: store ptr %{{[^,]+}}, ptr [[HANDLEADDR:%[^,]+]]
// CK4: store ptr %{{[^,]+}}, ptr [[BPTRADDR:%[^,]+]]
// CK4: store ptr %{{[^,]+}}, ptr [[VPTRADDR:%[^,]+]]
// CK4: store i64 %{{[^,]+}}, ptr [[SIZEADDR:%[^,]+]]
// CK4: store i64 %{{[^,]+}}, ptr [[TYPEADDR:%[^,]+]]
// CK4-DAG: [[BYTESIZE:%.+]] = load i64, ptr [[SIZEADDR]]
// CK4-64-DAG: [[SIZE:%.+]] = udiv exact i64 [[BYTESIZE]], 16
// CK4-32-DAG: [[SIZE:%.+]] = udiv exact i64 [[BYTESIZE]], 8
// CK4-DAG: [[TYPE:%.+]] = load i64, ptr [[TYPEADDR]]
// CK4-DAG: [[HANDLE:%.+]] = load ptr, ptr [[HANDLEADDR]]
// CK4-DAG: [[BPTR:%.+]] = load ptr, ptr [[BPTRADDR]]
// CK4-DAG: [[BEGIN:%.+]] = load ptr, ptr [[VPTRADDR]]
// CK4-DAG: [[PTREND:%.+]] = getelementptr %class.C, ptr [[BEGIN]], i64 [[SIZE]]
// CK4-DAG: [[ISARRAY:%.+]] = icmp sgt i64 [[SIZE]], 1
// CK4-DAG: [[PTRSNE:%.+]] = icmp ne ptr [[BPTR]], [[BEGIN]]
// CK4-DAG: [[PTRANDOBJ:%.+]] = and i64 [[TYPE]], 16
// CK4-DAG: [[ISPTRANDOBJ:%.+]] = icmp ne i64 [[PTRANDOBJ]], 0
// CK4-DAG: [[CMPA:%.+]] = and i1 [[PTRSNE]], [[ISPTRANDOBJ]]
// CK4-DAG: [[CMP:%.+]] = or i1 [[ISARRAY]], [[CMPA]]
// CK4-DAG: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK4-DAG: [[ISNOTDEL:%.+]] = icmp eq i64 [[TYPEDEL]], 0
// CK4-DAG: [[CMP1:%.+]] = and i1 [[CMP]], [[ISNOTDEL]]
// CK4: br i1 [[CMP1]], label %[[INITEVALDEL:[^,]+]], label %[[LHEAD:[^,]+]]

// CK4: [[INITEVALDEL]]
// CK4-64-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 16
// CK4-32-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 8

// Remove movement mappings and mark as implicit
// CK4-DAG: [[ITYPE:%.+]] = and i64 [[TYPE]], -4
// CK4-DAG: [[ITYPE1:%.+]] = or i64 [[ITYPE]], 512
// CK4: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[BPTR]], ptr [[BEGIN]], i64 [[ARRSIZE]], i64 [[ITYPE1]], {{.*}})
// CK4: br label %[[LHEAD:[^,]+]]

// CK4: [[LHEAD]]
// CK4: [[ISEMPTY:%.+]] = icmp eq ptr [[BEGIN]], [[PTREND]]
// CK4: br i1 [[ISEMPTY]], label %[[DONE:[^,]+]], label %[[LBODY:[^,]+]]
// CK4: [[LBODY]]
// CK4: [[PTR:%.+]] = phi ptr [ [[BEGIN]], %{{.+}} ], [ [[PTRNEXT:%.+]], %[[LCORRECT:[^,]+]] ]
// CK4-DAG: [[ABEGIN:%.+]] = getelementptr inbounds nuw %class.C, ptr [[PTR]], i32 0, i32 0
// CK4-DAG: [[BBEGIN:%.+]] = getelementptr inbounds nuw %class.C, ptr [[PTR]], i32 0, i32 1
// CK4-DAG: [[BBEGIN2:%.+]] = getelementptr inbounds nuw %class.C, ptr [[PTR]], i32 0, i32 1
// CK4-DAG: [[BARRBEGIN:%.+]] = load ptr, ptr [[BBEGIN2]]
// CK4-DAG: [[BARRBEGINGEP:%.+]] = getelementptr inbounds double, ptr [[BARRBEGIN]], i[[sz:64|32]] 0
// CK4-DAG: [[BEND:%.+]] = getelementptr ptr, ptr [[BBEGIN]], i32 1
// CK4-DAG: [[ABEGINI:%.+]] = ptrtoint ptr [[ABEGIN]] to i64
// CK4-DAG: [[BENDI:%.+]] = ptrtoint ptr [[BEND]] to i64
// CK4-DAG: [[CSIZE:%.+]] = sub i64 [[BENDI]], [[ABEGINI]]
// CK4-DAG: [[CUSIZE:%.+]] = sdiv exact i64 [[CSIZE]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CK4-DAG: [[PRESIZE:%.+]] = call i64 @__tgt_mapper_num_components(ptr [[HANDLE]])
// CK4-DAG: [[SHIPRESIZE:%.+]] = shl i64 [[PRESIZE]], 48
// CK4-DAG: [[MEMBERTYPE:%.+]] = add nuw i64 0, [[SHIPRESIZE]]
// CK4-DAG: [[TYPETF:%.+]] = and i64 [[TYPE]], 3
// CK4-DAG: [[ISALLOC:%.+]] = icmp eq i64 [[TYPETF]], 0
// CK4-DAG: br i1 [[ISALLOC]], label %[[ALLOC:[^,]+]], label %[[ALLOCELSE:[^,]+]]
// CK4-DAG: [[ALLOC]]
// CK4-DAG: [[ALLOCTYPE:%.+]] = and i64 [[MEMBERTYPE]], -4
// CK4-DAG: br label %[[TYEND:[^,]+]]
// CK4-DAG: [[ALLOCELSE]]
// CK4-DAG: [[ISTO:%.+]] = icmp eq i64 [[TYPETF]], 1
// CK4-DAG: br i1 [[ISTO]], label %[[TO:[^,]+]], label %[[TOELSE:[^,]+]]
// CK4-DAG: [[TO]]
// CK4-DAG: [[TOTYPE:%.+]] = and i64 [[MEMBERTYPE]], -3
// CK4-DAG: br label %[[TYEND]]
// CK4-DAG: [[TOELSE]]
// CK4-DAG: [[ISFROM:%.+]] = icmp eq i64 [[TYPETF]], 2
// CK4-DAG: br i1 [[ISFROM]], label %[[FROM:[^,]+]], label %[[TYEND]]
// CK4-DAG: [[FROM]]
// CK4-DAG: [[FROMTYPE:%.+]] = and i64 [[MEMBERTYPE]], -2
// CK4-DAG: br label %[[TYEND]]
// CK4-DAG: [[TYEND]]
// CK4-DAG: [[PHITYPE0:%.+]] = phi i64 [ [[ALLOCTYPE]], %[[ALLOC]] ], [ [[TOTYPE]], %[[TO]] ], [ [[FROMTYPE]], %[[FROM]] ], [ [[MEMBERTYPE]], %[[TOELSE]] ]
// CK4: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[PTR]], ptr [[ABEGIN]], i64 [[CUSIZE]], i64 [[PHITYPE0]], {{.*}})
// 281474976710659 == 0x1,000,000,003
// CK4-DAG: [[MEMBERTYPE:%.+]] = add nuw i64 281474976710659, [[SHIPRESIZE]]
// CK4-DAG: [[TYPETF:%.+]] = and i64 [[TYPE]], 3
// CK4-DAG: [[ISALLOC:%.+]] = icmp eq i64 [[TYPETF]], 0
// CK4-DAG: br i1 [[ISALLOC]], label %[[ALLOC:[^,]+]], label %[[ALLOCELSE:[^,]+]]
// CK4-DAG: [[ALLOC]]
// CK4-DAG: [[ALLOCTYPE:%.+]] = and i64 [[MEMBERTYPE]], -4
// CK4-DAG: br label %[[TYEND:[^,]+]]
// CK4-DAG: [[ALLOCELSE]]
// CK4-DAG: [[ISTO:%.+]] = icmp eq i64 [[TYPETF]], 1
// CK4-DAG: br i1 [[ISTO]], label %[[TO:[^,]+]], label %[[TOELSE:[^,]+]]
// CK4-DAG: [[TO]]
// CK4-DAG: [[TOTYPE:%.+]] = and i64 [[MEMBERTYPE]], -3
// CK4-DAG: br label %[[TYEND]]
// CK4-DAG: [[TOELSE]]
// CK4-DAG: [[ISFROM:%.+]] = icmp eq i64 [[TYPETF]], 2
// CK4-DAG: br i1 [[ISFROM]], label %[[FROM:[^,]+]], label %[[TYEND]]
// CK4-DAG: [[FROM]]
// CK4-DAG: [[FROMTYPE:%.+]] = and i64 [[MEMBERTYPE]], -2
// CK4-DAG: br label %[[TYEND]]
// CK4-DAG: [[TYEND]]
// CK4-DAG: [[TYPE1:%.+]] = phi i64 [ [[ALLOCTYPE]], %[[ALLOC]] ], [ [[TOTYPE]], %[[TO]] ], [ [[FROMTYPE]], %[[FROM]] ], [ [[MEMBERTYPE]], %[[TOELSE]] ]
// CK4: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[PTR]], ptr [[ABEGIN]], i64 4, i64 [[TYPE1]], {{.*}})
// 281474976710675 == 0x1,000,000,013
// CK4-DAG: [[MEMBERTYPE:%.+]] = add nuw i64 281474976710675, [[SHIPRESIZE]]
// CK4-DAG: [[TYPETF:%.+]] = and i64 [[TYPE]], 3
// CK4-DAG: [[ISALLOC:%.+]] = icmp eq i64 [[TYPETF]], 0
// CK4-DAG: br i1 [[ISALLOC]], label %[[ALLOC:[^,]+]], label %[[ALLOCELSE:[^,]+]]
// CK4-DAG: [[ALLOC]]
// CK4-DAG: [[ALLOCTYPE:%.+]] = and i64 [[MEMBERTYPE]], -4
// CK4-DAG: br label %[[TYEND:[^,]+]]
// CK4-DAG: [[ALLOCELSE]]
// CK4-DAG: [[ISTO:%.+]] = icmp eq i64 [[TYPETF]], 1
// CK4-DAG: br i1 [[ISTO]], label %[[TO:[^,]+]], label %[[TOELSE:[^,]+]]
// CK4-DAG: [[TO]]
// CK4-DAG: [[TOTYPE:%.+]] = and i64 [[MEMBERTYPE]], -3
// CK4-DAG: br label %[[TYEND]]
// CK4-DAG: [[TOELSE]]
// CK4-DAG: [[ISFROM:%.+]] = icmp eq i64 [[TYPETF]], 2
// CK4-DAG: br i1 [[ISFROM]], label %[[FROM:[^,]+]], label %[[TYEND]]
// CK4-DAG: [[FROM]]
// CK4-DAG: [[FROMTYPE:%.+]] = and i64 [[MEMBERTYPE]], -2
// CK4-DAG: br label %[[TYEND]]
// CK4-DAG: [[TYEND]]
// CK4-DAG: [[TYPE2:%.+]] = phi i64 [ [[ALLOCTYPE]], %[[ALLOC]] ], [ [[TOTYPE]], %[[TO]] ], [ [[FROMTYPE]], %[[FROM]] ], [ [[MEMBERTYPE]], %[[TOELSE]] ]
// CK4: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[BBEGIN]], ptr [[BARRBEGINGEP]], i64 16, i64 [[TYPE2]], {{.*}})
// CK4: [[PTRNEXT]] = getelementptr %class.C, ptr [[PTR]], i32 1
// CK4: [[ISDONE:%.+]] = icmp eq ptr [[PTRNEXT]], [[PTREND]]
// CK4: br i1 [[ISDONE]], label %[[LEXIT:[^,]+]], label %[[LBODY]]

// CK4: [[LEXIT]]
// CK4: [[ISARRAY:%.+]] = icmp sgt i64 [[SIZE]], 1
// CK4: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK4: [[ISNOTDEL:%.+]] = icmp ne i64 [[TYPEDEL]], 0
// CK4: [[CMP1:%.+]] = and i1 [[ISARRAY]], [[ISNOTDEL]]
// CK4: br i1 [[CMP1]], label %[[EVALDEL:[^,]+]], label %[[DONE]]
// CK4: [[EVALDEL]]
// CK4-64-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 16
// CK4-32-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 8

// Remove movement mappings and mark as implicit
// CK4-DAG: [[DTYPE:%.+]] = and i64 [[TYPE]], -4
// CK4-DAG: [[DTYPE1:%.+]] = or i64 [[DTYPE]], 512
// CK4: call void @__tgt_push_mapper_component(ptr [[HANDLE]], ptr [[BPTR]], ptr [[BEGIN]], i64 [[ARRSIZE]], i64 [[DTYPE1]], {{.*}})
// CK4: br label %[[DONE]]
// CK4: [[DONE]]
// CK4: ret void


// CK4-LABEL: define {{.*}}void @{{.*}}foo{{.*}}
void foo(int a){
  int i = a;
  C c;
  c.a = a;

  // CK4-DAG: call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[TGEPBP:%.+]], ptr [[TGEPP:%.+]], ptr [[TSIZES]], ptr [[TTYPES]], ptr null, ptr [[TMPR:%.+]])
  // CK4-DAG: [[TGEPBP]] = getelementptr inbounds {{.+}}[[TBP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[TGEPP]] = getelementptr inbounds {{.+}}[[TP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[TBP0:%.+]] = getelementptr inbounds {{.+}}[[TBP]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[TP0:%.+]] = getelementptr inbounds {{.+}}[[TP]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[TMPR1:%.+]] = getelementptr inbounds {{.+}}[[TMPR]], i[[sz]] 0, i[[sz]] 0
  // CK4-DAG: store ptr [[VAL:%[^,]+]], ptr [[TBP0]]
  // CK4-DAG: store ptr [[VAL]], ptr [[TP0]]
  // CK4-DAG: store ptr [[MPRFUNC]], ptr [[TMPR1]]
  #pragma omp target update to(present, mapper(id): c)

  // CK4-DAG: call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[FGEPBP:%.+]], ptr [[FGEPP:%.+]], ptr [[FSIZES]], ptr [[FTYPES]], ptr null, ptr [[FMPR:%.+]])
  // CK4-DAG: [[FGEPBP]] = getelementptr inbounds {{.+}}[[FBP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[FGEPP]] = getelementptr inbounds {{.+}}[[FP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[FBP0:%.+]] = getelementptr inbounds {{.+}}[[FBP]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[FP0:%.+]] = getelementptr inbounds {{.+}}[[FP]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[FMPR1:%.+]] = getelementptr inbounds {{.+}}[[FMPR]], i[[sz]] 0, i[[sz]] 0
  // CK4-DAG: store ptr [[VAL]], ptr [[FBP0]]
  // CK4-DAG: store ptr [[VAL]], ptr [[FP0]]
  // CK4-DAG: store ptr [[MPRFUNC]], ptr [[FMPR1]]
  #pragma omp target update from(mapper(id), present: c)
}

#endif // CK4

#endif // HEADER
