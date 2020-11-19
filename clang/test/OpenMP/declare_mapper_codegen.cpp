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

// CK0: [[IDENT_T:%.+]] = type { i32, i32, i32, i32, i8* }
// CK0: [[ENTRY:%.+]] = type { i8*, i8*, i[[SZ:32|64]], i32, i32 }
// CK0: [[ANON_T:%.+]] = type { %class.C* }
// CK0: [[ANON_T_0:%.+]] = type { %class.C* }
// CK0: [[KMP_TASK_T_WITH_PRIVATES:%.+]] = type { [[KMP_TASK_T:%[^,]+]], [[KMP_PRIVATES_T:%.+]] }
// CK0: [[KMP_TASK_T]] = type { i8*, i32 (i32, i8*)*, i32, %{{[^,]+}}, %{{[^,]+}} }
// CK0-32: [[KMP_PRIVATES_T]] = type { [1 x i64], [1 x i8*], [1 x i8*], [1 x i8*] }
// CK0-64: [[KMP_PRIVATES_T]] = type { [1 x i8*], [1 x i8*], [1 x i64], [1 x i8*] }
// CK0: [[KMP_TASK_T_WITH_PRIVATES_1:%.+]] = type { [[KMP_TASK_T]], [[KMP_PRIVATES_T_2:%.+]] }
// CK0-32: [[KMP_PRIVATES_T_2]] = type { [1 x i64], [1 x i8*], [1 x i8*], [1 x i8*] }
// CK0-64: [[KMP_PRIVATES_T_2]] = type { [1 x i8*], [1 x i8*], [1 x i64], [1 x i8*] }
// CK0: [[KMP_TASK_T_WITH_PRIVATES_4:%.+]] = type { [[KMP_TASK_T]], [[KMP_PRIVATES_T_5:%.+]] }
// CK0-32: [[KMP_PRIVATES_T_5]] = type { [1 x i64], [1 x i8*], [1 x i8*], [1 x i8*] }
// CK0-64: [[KMP_PRIVATES_T_5]] = type { [1 x i8*], [1 x i8*], [1 x i64], [1 x i8*] }
// CK0: [[KMP_TASK_T_WITH_PRIVATES_7:%.+]] = type { [[KMP_TASK_T]], [[KMP_PRIVATES_T_8:%.+]] }
// CK0-32: [[KMP_PRIVATES_T_8]] = type { [1 x i64], [1 x i8*], [1 x i8*], [1 x i8*] }
// CK0-64: [[KMP_PRIVATES_T_8]] = type { [1 x i8*], [1 x i8*], [1 x i64], [1 x i8*] }
// CK0: [[KMP_TASK_T_WITH_PRIVATES_10:%.+]] = type { [[KMP_TASK_T]], [[KMP_PRIVATES_T_11:%.+]] }
// CK0-32: [[KMP_PRIVATES_T_11]] = type { [1 x i64], [1 x i8*], [1 x i8*], [1 x i8*] }
// CK0-64: [[KMP_PRIVATES_T_11]] = type { [1 x i8*], [1 x i8*], [1 x i64], [1 x i8*] }

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
// CK0: [[EDTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 33]
// CK0-64: [[EDNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[EDNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[EDNWTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 33]
// CK0-64: [[EXDSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[EXDSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[EXDTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 34]
// CK0-64: [[EXDNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[EXDNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[EXDNWTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 34]
// CK0-64: [[TSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[TSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[TTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 33]
// CK0-64: [[FSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[FSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[FTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 34]
// CK0-64: [[FNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[FNWSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[FNWTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 34]

class C {
public:
  int a;
  double *b;
};

#pragma omp declare mapper(id: C s) map(s.a, s.b[0:2])

// CK0: define {{.*}}void [[MPRFUNC:@[.]omp_mapper[.].*C[.]id]](i8*{{.*}}, i8*{{.*}}, i8*{{.*}}, i64{{.*}}, i64{{.*}})
// CK0: store i8* %{{[^,]+}}, i8** [[HANDLEADDR:%[^,]+]]
// CK0: store i8* %{{[^,]+}}, i8** [[BPTRADDR:%[^,]+]]
// CK0: store i8* %{{[^,]+}}, i8** [[VPTRADDR:%[^,]+]]
// CK0: store i64 %{{[^,]+}}, i{{64|32}}* [[SIZEADDR:%[^,]+]]
// CK0: store i64 %{{[^,]+}}, i64* [[TYPEADDR:%[^,]+]]
// CK0-DAG: [[BYTESIZE:%.+]] = load i64, i64* [[SIZEADDR]]
// CK0-64-DAG: [[SIZE:%.+]] = udiv exact i64 [[BYTESIZE]], 16
// CK0-32-DAG: [[SIZE:%.+]] = udiv exact i64 [[BYTESIZE]], 8
// CK0-DAG: [[TYPE:%.+]] = load i64, i64* [[TYPEADDR]]
// CK0-DAG: [[HANDLE:%.+]] = load i8*, i8** [[HANDLEADDR]]
// CK0-DAG: [[PTRBEGIN:%.+]] = bitcast i8** [[VPTRADDR]] to %class.C**
// CK0-DAG: [[PTREND:%.+]] = getelementptr %class.C*, %class.C** [[PTRBEGIN]], i64 [[SIZE]]
// CK0-DAG: [[BPTR:%.+]] = load i8*, i8** [[BPTRADDR]]
// CK0-DAG: [[BEGIN:%.+]] = load i8*, i8** [[VPTRADDR]]
// CK0: [[ISARRAY:%.+]] = icmp sge i64 [[SIZE]], 1
// CK0: br i1 [[ISARRAY]], label %[[INITEVALDEL:[^,]+]], label %[[LHEAD:[^,]+]]

// CK0: [[INITEVALDEL]]
// CK0: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK0: [[ISNOTDEL:%.+]] = icmp eq i64 [[TYPEDEL]], 0
// CK0: br i1 [[ISNOTDEL]], label %[[INIT:[^,]+]], label %[[LHEAD:[^,]+]]
// CK0: [[INIT]]
// CK0-64-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 16
// CK0-32-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 8
// CK0-DAG: [[ITYPE:%.+]] = and i64 [[TYPE]], -4
// CK0: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTR]], i8* [[BEGIN]], i64 [[ARRSIZE]], i64 [[ITYPE]])
// CK0: br label %[[LHEAD:[^,]+]]

// CK0: [[LHEAD]]
// CK0: [[ISEMPTY:%.+]] = icmp eq %class.C** [[PTRBEGIN]], [[PTREND]]
// CK0: br i1 [[ISEMPTY]], label %[[DONE:[^,]+]], label %[[LBODY:[^,]+]]
// CK0: [[LBODY]]
// CK0: [[PTR:%.+]] = phi %class.C** [ [[PTRBEGIN]], %[[LHEAD]] ], [ [[PTRNEXT:%.+]], %[[LCORRECT:[^,]+]] ]
// CK0: [[OBJ:%.+]] = load %class.C*, %class.C** [[PTR]]
// CK0-DAG: [[ABEGIN:%.+]] = getelementptr inbounds %class.C, %class.C* [[OBJ]], i32 0, i32 0
// CK0-DAG: [[BBEGIN:%.+]] = getelementptr inbounds %class.C, %class.C* [[OBJ]], i32 0, i32 1
// CK0-DAG: [[BBEGIN2:%.+]] = getelementptr inbounds %class.C, %class.C* [[OBJ]], i32 0, i32 1
// CK0-DAG: [[BARRBEGIN:%.+]] = load double*, double** [[BBEGIN2]]
// CK0-DAG: [[BARRBEGINGEP:%.+]] = getelementptr inbounds double, double* [[BARRBEGIN]], i[[sz:64|32]] 0
// CK0-DAG: [[BEND:%.+]] = getelementptr double*, double** [[BBEGIN]], i32 1
// CK0-DAG: [[ABEGINV:%.+]] = bitcast i32* [[ABEGIN]] to i8*
// CK0-DAG: [[BENDV:%.+]] = bitcast double** [[BEND]] to i8*
// CK0-DAG: [[ABEGINI:%.+]] = ptrtoint i8* [[ABEGINV]] to i64
// CK0-DAG: [[BENDI:%.+]] = ptrtoint i8* [[BENDV]] to i64
// CK0-DAG: [[CSIZE:%.+]] = sub i64 [[BENDI]], [[ABEGINI]]
// CK0-DAG: [[CUSIZE:%.+]] = sdiv exact i64 [[CSIZE]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
// CK0-DAG: [[BPTRADDR0BC:%.+]] = bitcast %class.C* [[OBJ]] to i8*
// CK0-DAG: [[PTRADDR0BC:%.+]] = bitcast i32* [[ABEGIN]] to i8*
// CK0-DAG: [[PRESIZE:%.+]] = call i64 @__tgt_mapper_num_components(i8* [[HANDLE]])
// CK0-DAG: [[SHIPRESIZE:%.+]] = shl i64 [[PRESIZE]], 48
// CK0-DAG: br label %[[MEMBER:[^,]+]]
// CK0-DAG: [[MEMBER]]
// CK0-DAG: br i1 true, label %[[LTYPE:[^,]+]], label %[[MEMBERCOM:[^,]+]]
// CK0-DAG: [[MEMBERCOM]]
// CK0-DAG: [[MEMBERCOMTYPE:%.+]] = add nuw i64 32, [[SHIPRESIZE]]
// CK0-DAG: br label %[[LTYPE]]
// CK0-DAG: [[LTYPE]]
// CK0-DAG: [[MEMBERTYPE:%.+]] = phi i64 [ 32, %[[MEMBER]] ], [ [[MEMBERCOMTYPE]], %[[MEMBERCOM]] ]
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
// CK0: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTRADDR0BC]], i8* [[PTRADDR0BC]], i64 [[CUSIZE]], i64 [[PHITYPE0]])
// CK0-DAG: [[BPTRADDR1BC:%.+]] = bitcast %class.C* [[OBJ]] to i8*
// CK0-DAG: [[PTRADDR1BC:%.+]] = bitcast i32* [[ABEGIN]] to i8*
// CK0-DAG: br label %[[MEMBER:[^,]+]]
// CK0-DAG: [[MEMBER]]
// CK0-DAG: br i1 false, label %[[LTYPE:[^,]+]], label %[[MEMBERCOM:[^,]+]]
// CK0-DAG: [[MEMBERCOM]]
// 281474976710659 == 0x1,000,000,003
// CK0-DAG: [[MEMBERCOMTYPE:%.+]] = add nuw i64 281474976710659, [[SHIPRESIZE]]
// CK0-DAG: br label %[[LTYPE]]
// CK0-DAG: [[LTYPE]]
// CK0-DAG: [[MEMBERTYPE:%.+]] = phi i64 [ 281474976710659, %[[MEMBER]] ], [ [[MEMBERCOMTYPE]], %[[MEMBERCOM]] ]
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
// CK0: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTRADDR1BC]], i8* [[PTRADDR1BC]], i64 4, i64 [[TYPE1]])
// CK0-DAG: [[BPTRADDR2BC:%.+]] = bitcast double** [[BBEGIN]] to i8*
// CK0-DAG: [[PTRADDR2BC:%.+]] = bitcast double* [[BARRBEGINGEP]] to i8*
// CK0-DAG: br label %[[MEMBER:[^,]+]]
// CK0-DAG: [[MEMBER]]
// CK0-DAG: br i1 false, label %[[LTYPE:[^,]+]], label %[[MEMBERCOM:[^,]+]]
// CK0-DAG: [[MEMBERCOM]]
// 281474976710675 == 0x1,000,000,013
// CK0-DAG: [[MEMBERCOMTYPE:%.+]] = add nuw i64 281474976710675, [[SHIPRESIZE]]
// CK0-DAG: br label %[[LTYPE]]
// CK0-DAG: [[LTYPE]]
// CK0-DAG: [[MEMBERTYPE:%.+]] = phi i64 [ 281474976710675, %[[MEMBER]] ], [ [[MEMBERCOMTYPE]], %[[MEMBERCOM]] ]
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
// CK0: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTRADDR2BC]], i8* [[PTRADDR2BC]], i64 16, i64 [[TYPE2]])
// CK0: [[PTRNEXT]] = getelementptr %class.C*, %class.C** [[PTR]], i32 1
// CK0: [[ISDONE:%.+]] = icmp eq %class.C** [[PTRNEXT]], [[PTREND]]
// CK0: br i1 [[ISDONE]], label %[[LEXIT:[^,]+]], label %[[LBODY]]

// CK0: [[LEXIT]]
// CK0: [[ISARRAY:%.+]] = icmp sge i64 [[SIZE]], 1
// CK0: br i1 [[ISARRAY]], label %[[EVALDEL:[^,]+]], label %[[DONE]]
// CK0: [[EVALDEL]]
// CK0: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK0: [[ISDEL:%.+]] = icmp ne i64 [[TYPEDEL]], 0
// CK0: br i1 [[ISDEL]], label %[[DEL:[^,]+]], label %[[DONE]]
// CK0: [[DEL]]
// CK0-64-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 16
// CK0-32-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 8
// CK0-DAG: [[DTYPE:%.+]] = and i64 [[TYPE]], -4
// CK0: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTR]], i8* [[BEGIN]], i64 [[ARRSIZE]], i64 [[DTYPE]])
// CK0: br label %[[DONE]]
// CK0: [[DONE]]
// CK0: ret void


// CK0-LABEL: define {{.*}}void @{{.*}}foo{{.*}}
void foo(int a){
  int i = a;
  C c;
  c.a = a;

  // CK0-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null, i8** [[MPRGEP:%.+]])
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[MPRGEP]] = bitcast [1 x i8*]* [[MPR:%[^,]+]] to i8**
  // CK0-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK0-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK0-DAG: [[MPR1:%.+]] = getelementptr inbounds {{.+}}[[MPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to %class.C**
  // CK0-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to %class.C**
  // CK0-DAG: store %class.C* [[VAL:%[^,]+]], %class.C** [[CBP1]]
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[CP1]]
  // CK0-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[MPR1]]
  // CK0: call void [[KERNEL_1:@.+]](%class.C* [[VAL]])
  #pragma omp target map(mapper(id),tofrom: c)
  {
    ++c.a;
  }

  // CK0: [[BP2GEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[OFFLOAD_BP2:%[^,]+]], i32 0, i32 0
  // CK0: [[BP2CAST:%.+]] = bitcast i8** [[BP2GEP]] to %class.C**
  // CK0: store %class.C* [[CADDR:%[^,]+]], %class.C** [[BP2CAST]], align
  // CK0: [[P2GEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[OFFLOAD_P2:%[^,]+]], i32 0, i32 0
  // CK0: [[P2CAST:%.+]] = bitcast i8** [[P2GEP]] to %class.C**
  // CK0: store %class.C* [[CADDR]], %class.C** [[P2CAST]], align
  // CK0: [[MAPPER2GEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[OFFLOAD_MAPPER2:%[^,]+]], i[[SZ]] 0, i[[SZ]] 0
  // CK0: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[MAPPER2GEP]], align
  // CK0: [[BP2:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[OFFLOAD_BP2]], i32 0, i32 0
  // CK0: [[P2:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[OFFLOAD_P2]], i32 0, i32 0
  // CK0: [[MAPPER2:%.+]] = bitcast [1 x i8*]* [[OFFLOAD_MAPPER2]] to i8**
  // CK0-32: [[TASK:%.+]] = call i8* @__kmpc_omp_target_task_alloc([[IDENT_T]]* {{@.+}}, i32 {{%.+}}, i32 1, i32 40, i32 4, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T_WITH_PRIVATES]]*)* [[TASK_ENTRY:@.+]] to i32 (i32, i8*)*), i64 -1)
  // CK0-64: [[TASK:%.+]] = call i8* @__kmpc_omp_target_task_alloc([[IDENT_T]]* {{@.+}}, i32 {{%.+}}, i32 1, i64 72, i64 8, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T_WITH_PRIVATES]]*)* [[TASK_ENTRY:@.+]] to i32 (i32, i8*)*), i64 -1)
  // CK0: [[TASK_CAST:%.+]] = bitcast i8* [[TASK]] to [[KMP_TASK_T_WITH_PRIVATES]]*
  // CK0: [[TASK_WITH_PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES]], [[KMP_TASK_T_WITH_PRIVATES]]* [[TASK_CAST]], i32 0, i32 1
  // CK0: {{.+}} = call i32 @__kmpc_omp_task([[IDENT_T]]* @1, i32 {{.+}}, i8* [[TASK]])
  #pragma omp target map(mapper(id),tofrom: c) nowait
  {
    ++c.a;
  }

  // CK0-DAG: call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[TEAMSIZES]]{{.+}}, {{.+}}[[TEAMTYPES]]{{.+}}, i8** null, i8** [[MPRGEP:%.+]], i32 0, i32 0)
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[MPRGEP]] = bitcast [1 x i8*]* [[MPR:%[^,]+]] to i8**
  // CK0-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK0-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK0-DAG: [[MPR1:%.+]] = getelementptr inbounds {{.+}}[[MPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to %class.C**
  // CK0-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to %class.C**
  // CK0-DAG: store %class.C* [[VAL:%[^,]+]], %class.C** [[CBP1]]
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[CP1]]
  // CK0-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[MPR1]]
  // CK0: call void [[KERNEL_3:@.+]](%class.C* [[VAL]])
  #pragma omp target teams map(mapper(id),to: c)
  {
    ++c.a;
  }

  // CK0-32: [[TASK_1:%.+]] = call i8* @__kmpc_omp_target_task_alloc([[IDENT_T]]* {{@.+}}, i32 {{%.+}}, i32 1, i32 40, i32 4, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T_WITH_PRIVATES_1]]*)* [[TASK_ENTRY_1:@.+]] to i32 (i32, i8*)*), i64 -1)
  // CK0-64: [[TASK_1:%.+]] = call i8* @__kmpc_omp_target_task_alloc([[IDENT_T]]* {{@.+}}, i32 {{%.+}}, i32 1, i64 72, i64 8, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T_WITH_PRIVATES_1]]*)* [[TASK_ENTRY_1:@.+]] to i32 (i32, i8*)*), i64 -1)
  // CK0: [[TASK_CAST_1:%.+]] = bitcast i8* [[TASK_1]] to [[KMP_TASK_T_WITH_PRIVATES_1]]*
  // CK0: [[TASK_CAST_GET_1:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES_1]], [[KMP_TASK_T_WITH_PRIVATES_1]]* [[TASK_CAST_1]], i32 0, i32 0
  // CK0: {{.+}} = getelementptr inbounds [[KMP_TASK_T]], [[KMP_TASK_T]]* [[TASK_CAST_GET_1]], i32 0, i32 0
  // CK0: {{.+}} = call i32 @__kmpc_omp_task([[IDENT_T]]* @1, i32 {{.+}}, i8* [[TASK_1]])
  #pragma omp target teams map(mapper(id),to: c) nowait
  {
    ++c.a;
  }

  // CK0-DAG: call void @__tgt_target_data_begin_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[EDSIZES]]{{.+}}, {{.+}}[[EDTYPES]]{{.+}}, i8** null, i8** [[MPRGEP:%.+]])
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[MPRGEP]] = bitcast [1 x i8*]* [[MPR:%[^,]+]] to i8**
  // CK0-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK0-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK0-DAG: [[MPR1:%.+]] = getelementptr inbounds {{.+}}[[MPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to %class.C**
  // CK0-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to %class.C**
  // CK0-DAG: store %class.C* [[VAL:%[^,]+]], %class.C** [[CBP1]]
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[CP1]]
  // CK0-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[MPR1]]
  #pragma omp target enter data map(mapper(id),to: c)

  // CK0-DAG: call i32 @__kmpc_omp_task([[IDENT_T]]* @{{[^,]+}}, i32 %{{[^,]+}}, i8* [[TASK_2:%.+]])
  // CK0-DAG: [[TASK_2]] = call i8* @__kmpc_omp_target_task_alloc([[IDENT_T]]* @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i[[sz]] {{40|72}}, i[[sz]] 1, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T_WITH_PRIVATES_4]]*)* [[OMP_TASK_ENTRY_18:@[^,]+]] to i32 (i32, i8*)*), i64 -1)
  // CK0-DAG: [[TASK_WITH_PRIVATES:%.+]] = bitcast i8* [[TASK_2]] to [[KMP_TASK_T_WITH_PRIVATES_4]]*
  // CK0-DAG: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES_4]], [[KMP_TASK_T_WITH_PRIVATES_4]]* [[TASK_WITH_PRIVATES]], i32 0, i32 1
  // CK0-32-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_5]], [[KMP_PRIVATES_T_5]]* [[PRIVATES]], i32 0, i32 1
  // CK0-64-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_5]], [[KMP_PRIVATES_T_5]]* [[PRIVATES]], i32 0, i32 0
  // CK0-DAG: [[FPBPADDR:%.+]] = bitcast [1 x i8*]* [[FPBPGEP]] to i8*
  // CK0-DAG: [[BPADDR:%.+]] = bitcast i8** [[BPGEP:%.+]] to i8*
  // CK0-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPBPADDR]], i8* align {{4|8}} [[BPADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP:%.+]], i32 0, i32 0
  // CK0-DAG: [[BPGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP]], i32 0, i32 0
  // CK0-DAG: [[BPADDR:%.+]] = bitcast i8** [[BPGEP]] to %class.C**
  // CK0-DAG: store %class.C* [[C:%[^,]+]], %class.C** [[BPADDR]], align
  // CK0-32-DAG: [[FPPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_5]], [[KMP_PRIVATES_T_5]]* [[PRIVATES]], i32 0, i32 2
  // CK0-64-DAG: [[FPPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_5]], [[KMP_PRIVATES_T_5]]* [[PRIVATES]], i32 0, i32 1
  // CK0-DAG: [[FPPADDR:%.+]] = bitcast [1 x i8*]* [[FPPGEP]] to i8*
  // CK0-DAG: [[PADDR:%.+]] = bitcast i8** [[PGEP:%.+]] to i8*
  // CK0-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPPADDR]], i8* align {{4|8}} [[PADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[PGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P:%.+]], i32 0, i32 0
  // CK0-DAG: [[PGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P]], i32 0, i32 0
  // CK0-DAG: [[PADDR:%.+]] = bitcast i8** [[PGEP]] to %class.C**
  // CK0-DAG: store %class.C* [[C]], %class.C** [[PADDR]], align
  // CK0-32-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_5]], [[KMP_PRIVATES_T_5]]* [[PRIVATES]], i32 0, i32 0
  // CK0-64-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_5]], [[KMP_PRIVATES_T_5]]* [[PRIVATES]], i32 0, i32 2
  // CK0-DAG: [[FPSZADDR:%.+]] = bitcast [1 x i64]* [[FPSZGEP]] to i8*
  // CK0-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPSZADDR]], i8* align {{4|8}} bitcast ([1 x i64]* [[EDNWSIZES]] to i8*), i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[FPMPRGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_5]], [[KMP_PRIVATES_T_5]]* [[PRIVATES]], i32 0, i32 3
  // CK0-DAG: [[FPMPRADDR:%.+]] = bitcast [1 x i8*]* [[FPMPRGEP]] to i8*
  // CK0-DAG: [[MPRADDR:%.+]] = bitcast i8** [[MPRGEP:%.+]] to i8*
  // CK0-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPMPRADDR]], i8* align {{4|8}} [[MPRADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[MPRGEP]] = bitcast [1 x i8*]* [[MPR:%.+]] to i8**
  // CK0-DAG: [[MPRGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[MPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[MPRGEP]], align
  #pragma omp target enter data map(mapper(id),to: c) nowait

  // CK0-DAG: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[EXDSIZES]]{{.+}}, {{.+}}[[EXDTYPES]]{{.+}}, i8** null, i8** [[MPRGEP:%.+]])
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[MPRGEP]] = bitcast [1 x i8*]* [[MPR:%[^,]+]] to i8**
  // CK0-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK0-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK0-DAG: [[MPR1:%.+]] = getelementptr inbounds {{.+}}[[MPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to %class.C**
  // CK0-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to %class.C**
  // CK0-DAG: store %class.C* [[VAL:%[^,]+]], %class.C** [[CBP1]]
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[CP1]]
  // CK0-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[MPR1]]
  #pragma omp target exit data map(mapper(id),from: c)

  // CK0-DAG: call i32 @__kmpc_omp_task([[IDENT_T]]* @{{[^,]+}}, i32 %{{[^,]+}}, i8* [[TASK_3:%.+]])
  // CK0-DAG: [[TASK_3]] = call i8* @__kmpc_omp_target_task_alloc([[IDENT_T]]* @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i[[sz]] {{40|72}}, i[[sz]] 1, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T_WITH_PRIVATES_7]]*)* [[OMP_TASK_ENTRY_25:@[^,]+]] to i32 (i32, i8*)*), i64 -1)
  // CK0-DAG: [[TASK_WITH_PRIVATES:%.+]] = bitcast i8* [[TASK_3]] to [[KMP_TASK_T_WITH_PRIVATES_7]]*
  // CK0-DAG: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES_7]], [[KMP_TASK_T_WITH_PRIVATES_7]]* [[TASK_WITH_PRIVATES]], i32 0, i32 1
  // CK0-32-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_8]], [[KMP_PRIVATES_T_8]]* [[PRIVATES]], i32 0, i32 1
  // CK0-64-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_8]], [[KMP_PRIVATES_T_8]]* [[PRIVATES]], i32 0, i32 0
  // CK0-DAG: [[FPBPADDR:%.+]] = bitcast [1 x i8*]* [[FPBPGEP]] to i8*
  // CK0-DAG: [[BPADDR:%.+]] = bitcast i8** [[BPGEP:%.+]] to i8*
  // CK0-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPBPADDR]], i8* align {{4|8}} [[BPADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP:%.+]], i32 0, i32 0
  // CK0-DAG: [[BPGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP]], i32 0, i32 0
  // CK0-DAG: [[BPADDR:%.+]] = bitcast i8** [[BPGEP]] to %class.C**
  // CK0-DAG: store %class.C* [[C:%[^,]+]], %class.C** [[BPADDR]], align
  // CK0-32-DAG: [[FPPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_8]], [[KMP_PRIVATES_T_8]]* [[PRIVATES]], i32 0, i32 2
  // CK0-64-DAG: [[FPPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_8]], [[KMP_PRIVATES_T_8]]* [[PRIVATES]], i32 0, i32 1
  // CK0-DAG: [[FPPADDR:%.+]] = bitcast [1 x i8*]* [[FPPGEP]] to i8*
  // CK0-DAG: [[PADDR:%.+]] = bitcast i8** [[PGEP:%.+]] to i8*
  // CK0-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPPADDR]], i8* align {{4|8}} [[PADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[PGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P:%.+]], i32 0, i32 0
  // CK0-DAG: [[PGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P]], i32 0, i32 0
  // CK0-DAG: [[PADDR:%.+]] = bitcast i8** [[PGEP]] to %class.C**
  // CK0-DAG: store %class.C* [[C]], %class.C** [[PADDR]], align
  // CK0-32-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_8]], [[KMP_PRIVATES_T_8]]* [[PRIVATES]], i32 0, i32 0
  // CK0-64-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_8]], [[KMP_PRIVATES_T_8]]* [[PRIVATES]], i32 0, i32 2
  // CK0-DAG: [[FPSZADDR:%.+]] = bitcast [1 x i64]* [[FPSZGEP]] to i8*
  // CK0-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPSZADDR]], i8* align {{4|8}} bitcast ([1 x i64]* [[EXDNWSIZES]] to i8*), i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[FPMPRGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_8]], [[KMP_PRIVATES_T_8]]* [[PRIVATES]], i32 0, i32 3
  // CK0-DAG: [[FPMPRADDR:%.+]] = bitcast [1 x i8*]* [[FPMPRGEP]] to i8*
  // CK0-DAG: [[MPRADDR:%.+]] = bitcast i8** [[MPRGEP:%.+]] to i8*
  // CK0-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPMPRADDR]], i8* align {{4|8}} [[MPRADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[MPRGEP]] = bitcast [1 x i8*]* [[MPR:%.+]] to i8**
  // CK0-DAG: [[MPRGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[MPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[MPRGEP]], align
  #pragma omp target exit data map(mapper(id),from: c) nowait

  // CK0-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[TGEPBP:%.+]], i8** [[TGEPP:%.+]], i64* getelementptr {{.+}}[1 x i64]* [[TSIZES]], i32 0, i32 0), {{.+}}getelementptr {{.+}}[1 x i64]* [[TTYPES]]{{.+}}, i8** null, i8** [[TMPRGEP:%.+]])
  // CK0-DAG: [[TGEPBP]] = getelementptr inbounds {{.+}}[[TBP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[TGEPP]] = getelementptr inbounds {{.+}}[[TP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[TMPRGEP]] = bitcast [1 x i8*]* [[TMPR:%[^,]+]] to i8**
  // CK0-DAG: [[TBP0:%.+]] = getelementptr inbounds {{.+}}[[TBP]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[TP0:%.+]] = getelementptr inbounds {{.+}}[[TP]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[TMPR1:%.+]] = getelementptr inbounds {{.+}}[[TMPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: [[TCBP0:%.+]] = bitcast i8** [[TBP0]] to %class.C**
  // CK0-DAG: [[TCP0:%.+]] = bitcast i8** [[TP0]] to %class.C**
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[TCBP0]]
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[TCP0]]
  // CK0-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[TMPR1]]
  #pragma omp target update to(mapper(id): c)

  // CK0-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[FGEPBP:%.+]], i8** [[FGEPP:%.+]], i64* getelementptr {{.+}}[1 x i64]* [[FSIZES]], i32 0, i32 0), {{.+}}getelementptr {{.+}}[1 x i64]* [[FTYPES]]{{.+}}, i8** null, i8** [[FMPRGEP:%.+]])
  // CK0-DAG: [[FGEPBP]] = getelementptr inbounds {{.+}}[[FBP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[FGEPP]] = getelementptr inbounds {{.+}}[[FP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[FMPRGEP]] = bitcast [1 x i8*]* [[FMPR:%[^,]+]] to i8**
  // CK0-DAG: [[FBP0:%.+]] = getelementptr inbounds {{.+}}[[FBP]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[FP0:%.+]] = getelementptr inbounds {{.+}}[[FP]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[FMPR1:%.+]] = getelementptr inbounds {{.+}}[[FMPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: [[FCBP0:%.+]] = bitcast i8** [[FBP0]] to %class.C**
  // CK0-DAG: [[FCP0:%.+]] = bitcast i8** [[FP0]] to %class.C**
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[FCBP0]]
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[FCP0]]
  // CK0-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[FMPR1]]
  #pragma omp target update from(mapper(id): c)

  // CK0-DAG: call i32 @__kmpc_omp_task([[IDENT_T]]* @{{[^,]+}}, i32 %{{[^,]+}}, i8* [[TASK_4:%.+]])
  // CK0-DAG: [[TASK_4]] = call i8* @__kmpc_omp_target_task_alloc([[IDENT_T]]* @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i[[sz]] {{40|72}}, i[[sz]] 1, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T_WITH_PRIVATES_10]]*)* [[OMP_TASK_ENTRY_34:@[^,]+]] to i32 (i32, i8*)*), i64 -1)
  // CK0-DAG: [[TASK_WITH_PRIVATES:%.+]] = bitcast i8* [[TASK_4]] to [[KMP_TASK_T_WITH_PRIVATES_10]]*
  // CK0-DAG: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES_10]], [[KMP_TASK_T_WITH_PRIVATES_10]]* [[TASK_WITH_PRIVATES]], i32 0, i32 1
  // CK0-32-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_11]], [[KMP_PRIVATES_T_11]]* [[PRIVATES]], i32 0, i32 1
  // CK0-64-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_11]], [[KMP_PRIVATES_T_11]]* [[PRIVATES]], i32 0, i32 0
  // CK0-DAG: [[FPBPADDR:%.+]] = bitcast [1 x i8*]* [[FPBPGEP]] to i8*
  // CK0-DAG: [[BPADDR:%.+]] = bitcast i8** [[BPGEP:%.+]] to i8*
  // CK0-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPBPADDR]], i8* align {{4|8}} [[BPADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP:%.+]], i32 0, i32 0
  // CK0-DAG: [[BPGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP]], i32 0, i32 0
  // CK0-DAG: [[BPADDR:%.+]] = bitcast i8** [[BPGEP]] to %class.C**
  // CK0-DAG: store %class.C* [[C:%[^,]+]], %class.C** [[BPADDR]], align
  // CK0-32-DAG: [[FPPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_11]], [[KMP_PRIVATES_T_11]]* [[PRIVATES]], i32 0, i32 2
  // CK0-64-DAG: [[FPPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_11]], [[KMP_PRIVATES_T_11]]* [[PRIVATES]], i32 0, i32 1
  // CK0-DAG: [[FPPADDR:%.+]] = bitcast [1 x i8*]* [[FPPGEP]] to i8*
  // CK0-DAG: [[PADDR:%.+]] = bitcast i8** [[PGEP:%.+]] to i8*
  // CK0-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPPADDR]], i8* align {{4|8}} [[PADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[PGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P:%.+]], i32 0, i32 0
  // CK0-DAG: [[PGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P]], i32 0, i32 0
  // CK0-DAG: [[PADDR:%.+]] = bitcast i8** [[PGEP]] to %class.C**
  // CK0-DAG: store %class.C* [[C]], %class.C** [[PADDR]], align
  // CK0-32-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_11]], [[KMP_PRIVATES_T_11]]* [[PRIVATES]], i32 0, i32 0
  // CK0-64-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_11]], [[KMP_PRIVATES_T_11]]* [[PRIVATES]], i32 0, i32 2
  // CK0-DAG: [[FPSZADDR:%.+]] = bitcast [1 x i64]* [[FPSZGEP]] to i8*
  // CK0-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPSZADDR]], i8* align {{4|8}} bitcast ([1 x i64]* [[FNWSIZES]] to i8*), i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[FPMPRGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T_11]], [[KMP_PRIVATES_T_11]]* [[PRIVATES]], i32 0, i32 3
  // CK0-DAG: [[FPMPRADDR:%.+]] = bitcast [1 x i8*]* [[FPMPRGEP]] to i8*
  // CK0-DAG: [[MPRADDR:%.+]] = bitcast i8** [[MPRGEP:%.+]] to i8*
  // CK0-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPMPRADDR]], i8* align {{4|8}} [[MPRADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK0-DAG: [[MPRGEP]] = bitcast [1 x i8*]* [[MPR:%.+]] to i8**
  // CK0-DAG: [[MPRGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[MPR]], i[[sz]] 0, i[[sz]] 0
  // CK0-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[MPRGEP]], align
  #pragma omp target update from(mapper(id): c) nowait
}


// CK0: define internal void [[KERNEL_1]](%class.C* {{.+}}[[ARG:%.+]])
// CK0: [[ADDR:%.+]] = alloca %class.C*,
// CK0: store %class.C* [[ARG]], %class.C** [[ADDR]]
// CK0: [[CADDR:%.+]] = load %class.C*, %class.C** [[ADDR]]
// CK0: [[CAADDR:%.+]] = getelementptr inbounds %class.C, %class.C* [[CADDR]], i32 0, i32 0
// CK0: [[VAL:%[^,]+]] = load i32, i32* [[CAADDR]]
// CK0: {{.+}} = add nsw i32 [[VAL]], 1
// CK0: }

// CK0: define internal void [[KERNEL_2:@.+]](%class.C* {{.+}}[[ARG:%.+]])
// CK0: [[ADDR:%.+]] = alloca %class.C*,
// CK0: store %class.C* [[ARG]], %class.C** [[ADDR]]
// CK0: [[CADDR:%.+]] = load %class.C*, %class.C** [[ADDR]]
// CK0: [[CAADDR:%.+]] = getelementptr inbounds %class.C, %class.C* [[CADDR]], i32 0, i32 0
// CK0: [[VAL:%[^,]+]] = load i32, i32* [[CAADDR]]
// CK0: {{.+}} = add nsw i32 [[VAL]], 1
// CK0: }

// CK0: define internal void [[OUTLINED:@.+]](i32 {{.*}}{{[^,]+}}, [[ANON_T]]* noalias [[CTXARG:%.+]])
// CK0-DAG: call i32 @__tgt_target_nowait_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZEGEP:%[0-9]+]], {{.+}}[[NWTYPES]]{{.+}}, i8** null, i8** [[MPRGEP:%.+]])
// CK0-DAG: [[BPGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[PGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[SIZEGEP]] = getelementptr inbounds [1 x i64], [1 x i64]* [[SIZEFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[MPRGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[MPRFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[BPFPADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPPTRADDR_BP:%.+]], align
// CK0-DAG: [[PFPADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPPTRADDR_P:%.+]], align
// CK0-DAG: [[SIZEFPADDR]] = load [1 x i64]*, [1 x i64]** [[FPPTRADDR_SIZE:%.+]], align
// CK0-DAG: [[MPRFPADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPPTRADDR_MPR:%.+]], align
// CK0-DAG: call void (i8*, ...) %1(i8* %2, {{.+}}[[FPPTRADDR_BP]], {{.+}}[[FPPTRADDR_P]], {{.+}}[[FPPTRADDR_SIZE]], {{.+}}[[FPPTRADDR_MPR]])
// CK0-DAG: call void [[KERNEL_2:@.+]](%class.C* [[KERNELARG:%.+]])
// CK0-DAG: [[KERNELARG]] = load %class.C*, %class.C** [[KERNELARGGEP:%.+]], align
// CK0-DAG: [[KERNELARGGEP]] = getelementptr inbounds [[ANON_T]], [[ANON_T]]* [[CTX:%.+]], i32 0, i32 0
// CK0-DAG: [[CTX]] = load [[ANON_T]]*, [[ANON_T]]** [[CTXADDR:%.+]], align
// CK0-DAG: store [[ANON_T]]* [[CTXARG]], [[ANON_T]]** [[CTXADDR]], align
// CK0: }

// CK0: define internal {{.*}}i32 [[TASK_ENTRY]](i32 {{.*}}%0, [[KMP_TASK_T_WITH_PRIVATES]]* noalias %1)
// CK0: store [[KMP_TASK_T_WITH_PRIVATES]]* %1, [[KMP_TASK_T_WITH_PRIVATES]]** [[ADDR:%.+]], align
// CK0: [[TASK_T_WITH_PRIVATES:%.+]] = load [[KMP_TASK_T_WITH_PRIVATES]]*, [[KMP_TASK_T_WITH_PRIVATES]]** [[ADDR]], align
// CK0: [[TASKGEP:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES]], [[KMP_TASK_T_WITH_PRIVATES]]* [[TASK_T_WITH_PRIVATES]], i32 0, i32 0
// CK0: [[SHAREDSGEP:%.+]] = getelementptr inbounds [[KMP_TASK_T]], [[KMP_TASK_T]]* [[TASKGEP]], i32 0, i32 0
// CK0: [[SHAREDS:%.+]] = load i8*, i8** [[SHAREDSGEP]], align
// CK0: [[ANON:%.+]] = bitcast i8* [[SHAREDS]] to [[ANON_T]]*
// CK0: [[PRIVATESGEP:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES]], [[KMP_TASK_T_WITH_PRIVATES]]* [[TASK_T_WITH_PRIVATES]], i32 0, i32 1
// CK0: [[PRIVATES:%.+]] = bitcast [[KMP_PRIVATES_T]]* [[PRIVATESGEP]] to i8*
// CK0: [[TASK_WITH_PRIVATES:%.+]] = bitcast [[KMP_TASK_T_WITH_PRIVATES]]* [[TASK_T_WITH_PRIVATES]] to i8*
// CK0: call void [[OUTLINED]](i32 {{%.+}}, i32* {{%.+}}, i8* [[PRIVATES]], {{.+}}, i8* [[TASK_WITH_PRIVATES]], [[ANON_T]]* [[ANON]])
// CK0: }

// CK0: define internal void [[OUTLINE_1:@.+]](i32 {{.*}}%.global_tid.{{.+}}, [[ANON_T_0]]* noalias [[CTXARG:%.+]])
// CK0-DAG: call i32 @__tgt_target_teams_nowait_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], i64* [[SIZEGEP:%[0-9]+]], {{.+}}[[TEAMNWTYPES]]{{.+}}, i8** null, i8** [[MPRGEP:%.+]], i32 0, i32 0)
// CK0-DAG: [[BPGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[PGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[SIZEGEP]] = getelementptr inbounds [1 x i64], [1 x i64]* [[SIZEFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[MPRGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[MPRFPADDR:%.+]], i[[SZ]] 0, i[[SZ]] 0
// CK0-DAG: [[BPFPADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPPTRADDR_BP:%.+]], align
// CK0-DAG: [[PFPADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPPTRADDR_P:%.+]], align
// CK0-DAG: [[SIZEFPADDR]] = load [1 x i64]*, [1 x i64]** [[FPPTRADDR_SIZE:%.+]], align
// CK0-DAG: [[MPRFPADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPPTRADDR_MPR:%.+]], align
// CK0-DAG: call void (i8*, ...) %1(i8* %2, {{.+}}[[FPPTRADDR_BP]], {{.+}}[[FPPTRADDR_P]], {{.+}}[[FPPTRADDR_SIZE]], {{.+}}[[FPPTRADDR_MPR]])
// CK0-DAG: call void [[KERNEL_2:@.+]](%class.C* [[KERNELARG:%.+]])
// CK0-DAG: [[KERNELARG]] = load %class.C*, %class.C** [[KERNELARGGEP:%.+]], align
// CK0-DAG: [[KERNELARGGEP]] = getelementptr inbounds [[ANON_T_0]], [[ANON_T_0]]* [[CTX:%.+]], i32 0, i32 0
// CK0-DAG: [[CTX]] = load [[ANON_T_0]]*, [[ANON_T_0]]** [[CTXADDR:%.+]], align
// CK0-DAG: store [[ANON_T_0]]* [[CTXARG]], [[ANON_T_0]]** [[CTXADDR]], align
// CK0: }

// CK0: define internal {{.*}}i32 [[TASK_ENTRY_1]](i32 {{.*}}%0, [[KMP_TASK_T_WITH_PRIVATES_1]]* noalias %1)
// CK0: store [[KMP_TASK_T_WITH_PRIVATES_1]]* %1, [[KMP_TASK_T_WITH_PRIVATES_1]]** [[ADDR:%.+]], align
// CK0: [[TASK_T_WITH_PRIVATES:%.+]] = load [[KMP_TASK_T_WITH_PRIVATES_1]]*, [[KMP_TASK_T_WITH_PRIVATES_1]]** [[ADDR]], align
// CK0: [[TASKGEP:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES_1]], [[KMP_TASK_T_WITH_PRIVATES_1]]* [[TASK_T_WITH_PRIVATES]], i32 0, i32 0
// CK0: [[SHAREDSGEP:%.+]] = getelementptr inbounds [[KMP_TASK_T]], [[KMP_TASK_T]]* [[TASKGEP]], i32 0, i32 0
// CK0: [[SHAREDS:%.+]] = load i8*, i8** [[SHAREDSGEP]], align
// CK0: [[ANON:%.+]] = bitcast i8* [[SHAREDS]] to [[ANON_T_0]]*
// CK0: [[PRIVATESGEP:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES_1]], [[KMP_TASK_T_WITH_PRIVATES_1]]* [[TASK_T_WITH_PRIVATES]], i32 0, i32 1
// CK0: [[PRIVATES:%.+]] = bitcast [[KMP_PRIVATES_T_2]]* [[PRIVATESGEP]] to i8*
// CK0: [[TASK_WITH_PRIVATES:%.+]] = bitcast [[KMP_TASK_T_WITH_PRIVATES_1]]* [[TASK_T_WITH_PRIVATES]] to i8*
// CK0: call void [[OUTLINE_1]](i32 {{%.+}}, i32* {{%.+}}, i8* [[PRIVATES]], {{.+}}, i8* [[TASK_WITH_PRIVATES]], [[ANON_T_0]]* [[ANON]])
// CK0: }

// CK0: define internal void [[OMP_OUTLINED_16:@.+]](i32{{.*}} %{{[^,]+}}, i32* noalias %{{[^,]+}}, i8* noalias %{{[^,]+}}
// CK0-DAG: call void @__tgt_target_data_begin_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[BP:%[^,]+]], i8** [[P:%[^,]+]], i64* [[SZ:%[^,]+]], i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[EDNWTYPES]], i32 0, i32 0), i8** null, i8** [[MPR:%.+]])
// CK0-DAG: [[BP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[P]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[SZ]] = getelementptr inbounds [1 x i64], [1 x i64]* [[SZADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[MPR]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[MPRADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[BPADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPBPADDR:%[^,]+]], align
// CK0-DAG: [[PADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPPADDR:%[^,]+]], align
// CK0-DAG: [[SZADDR]] = load [1 x i64]*, [1 x i64]** [[FPSZADDR:%[^,]+]], align
// CK0-DAG: [[MPRADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPMPRADDR:%[^,]+]], align
// CK0-DAG: call void (i8*, ...) %{{.+}}(i8* %{{[^,]+}}, [1 x i8*]** [[FPBPADDR]], [1 x i8*]** [[FPPADDR]], [1 x i64]** [[FPSZADDR]], [1 x i8*]** [[FPMPRADDR]])
// CK0: ret void
// CK0: }

// CK0: define internal {{.*}}i32 [[OMP_TASK_ENTRY_18]](i32 {{.*}}%{{[^,]+}}, [[KMP_TASK_T_WITH_PRIVATES_4]]* noalias %{{[^,]+}})
// CK0:   call void [[OMP_OUTLINED_16]]
// CK0:   ret i32 0
// CK0: }

// CK0: define internal void [[OMP_OUTLINED_23:@.+]](i32{{.*}} %{{[^,]+}}, i32* noalias %{{[^,]+}}, i8* noalias %{{[^,]+}}
// CK0-DAG: call void @__tgt_target_data_end_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[BP:%[^,]+]], i8** [[P:%[^,]+]], i64* [[SZ:%[^,]+]], i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[EXDNWTYPES]], i32 0, i32 0), i8** null, i8** [[MPR:%.+]])
// CK0-DAG: [[BP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[P]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[SZ]] = getelementptr inbounds [1 x i64], [1 x i64]* [[SZADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[MPR]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[MPRADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[BPADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPBPADDR:%[^,]+]], align
// CK0-DAG: [[PADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPPADDR:%[^,]+]], align
// CK0-DAG: [[SZADDR]] = load [1 x i64]*, [1 x i64]** [[FPSZADDR:%[^,]+]], align
// CK0-DAG: [[MPRADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPMPRADDR:%[^,]+]], align
// CK0-DAG: call void (i8*, ...) %{{.+}}(i8* %{{[^,]+}}, [1 x i8*]** [[FPBPADDR]], [1 x i8*]** [[FPPADDR]], [1 x i64]** [[FPSZADDR]], [1 x i8*]** [[FPMPRADDR]])
// CK0: }

// CK0: define internal {{.*}}i32 [[OMP_TASK_ENTRY_25]](i32 {{.*}}%{{[^,]+}}, [[KMP_TASK_T_WITH_PRIVATES_7]]* noalias %{{[^,]+}})
// CK0:   call void [[OMP_OUTLINED_23]]
// CK0:   ret i32 0
// CK0: }

// CK0: define internal void [[OMP_OUTLINED_32:@.+]](i32{{.*}} %{{[^,]+}}, i32* noalias %{{[^,]+}}, i8* noalias %{{[^,]+}}
// CK0-DAG: call void @__tgt_target_data_update_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[BP:%[^,]+]], i8** [[P:%[^,]+]], i64* [[SZ:%[^,]+]], i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[FNWTYPES]], i32 0, i32 0), i8** null, i8** [[MPR:%.+]])
// CK0-DAG: [[BP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[P]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[SZ]] = getelementptr inbounds [1 x i64], [1 x i64]* [[SZADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[MPR]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[MPRADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK0-DAG: [[BPADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPBPADDR:%[^,]+]], align
// CK0-DAG: [[PADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPPADDR:%[^,]+]], align
// CK0-DAG: [[SZADDR]] = load [1 x i64]*, [1 x i64]** [[FPSZADDR:%[^,]+]], align
// CK0-DAG: [[MPRADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPMPRADDR:%[^,]+]], align
// CK0-DAG: call void (i8*, ...) %{{.+}}(i8* %{{[^,]+}}, [1 x i8*]** [[FPBPADDR]], [1 x i8*]** [[FPPADDR]], [1 x i64]** [[FPSZADDR]], [1 x i8*]** [[FPMPRADDR]])
// CK0: }

// CK0: define internal {{.*}}i32 [[OMP_TASK_ENTRY_34]](i32 {{.*}}%{{[^,]+}}, [[KMP_TASK_T_WITH_PRIVATES_10]]* noalias %{{[^,]+}})
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

// CK1-LABEL: define {{.*}}void @.omp_mapper.{{.*}}C{{.*}}.id{{.*}}(i8*{{.*}}, i8*{{.*}}, i8*{{.*}}, i64{{.*}}, i64{{.*}})
// CK1: store i8* %{{[^,]+}}, i8** [[HANDLEADDR:%[^,]+]]
// CK1: store i8* %{{[^,]+}}, i8** [[BPTRADDR:%[^,]+]]
// CK1: store i8* %{{[^,]+}}, i8** [[VPTRADDR:%[^,]+]]
// CK1: store i64 %{{[^,]+}}, i{{64|32}}* [[SIZEADDR:%[^,]+]]
// CK1: store i64 %{{[^,]+}}, i64* [[TYPEADDR:%[^,]+]]
// CK1-DAG: [[BYTESIZE:%.+]] = load i64, i64* [[SIZEADDR]]
// CK1-DAG: [[SIZE:%.+]] = udiv exact i64 [[BYTESIZE]], 4
// CK1-DAG: [[TYPE:%.+]] = load i64, i64* [[TYPEADDR]]
// CK1-DAG: [[HANDLE:%.+]] = load i8*, i8** [[HANDLEADDR]]
// CK1-DAG: [[PTRBEGIN:%.+]] = bitcast i8** [[VPTRADDR]] to %class.C**
// CK1-DAG: [[PTREND:%.+]] = getelementptr %class.C*, %class.C** [[PTRBEGIN]], i64 [[SIZE]]
// CK1-DAG: [[BPTR:%.+]] = load i8*, i8** [[BPTRADDR]]
// CK1-DAG: [[BEGIN:%.+]] = load i8*, i8** [[VPTRADDR]]
// CK1: [[ISARRAY:%.+]] = icmp sge i64 [[SIZE]], 1
// CK1: br i1 [[ISARRAY]], label %[[INITEVALDEL:[^,]+]], label %[[LHEAD:[^,]+]]

// CK1: [[INITEVALDEL]]
// CK1: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK1: [[ISNOTDEL:%.+]] = icmp eq i64 [[TYPEDEL]], 0
// CK1: br i1 [[ISNOTDEL]], label %[[INIT:[^,]+]], label %[[LHEAD:[^,]+]]
// CK1: [[INIT]]
// CK1-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 4
// CK1-DAG: [[ITYPE:%.+]] = and i64 [[TYPE]], -4
// CK1: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTR]], i8* [[BEGIN]], i64 [[ARRSIZE]], i64 [[ITYPE]])
// CK1: br label %[[LHEAD:[^,]+]]

// CK1: [[LHEAD]]
// CK1: [[ISEMPTY:%.+]] = icmp eq %class.C** [[PTRBEGIN]], [[PTREND]]
// CK1: br i1 [[ISEMPTY]], label %[[DONE:[^,]+]], label %[[LBODY:[^,]+]]
// CK1: [[LBODY]]
// CK1: [[PTR:%.+]] = phi %class.C** [ [[PTRBEGIN]], %[[LHEAD]] ], [ [[PTRNEXT:%.+]], %[[LCORRECT:[^,]+]] ]
// CK1: [[OBJ:%.+]] = load %class.C*, %class.C** [[PTR]]
// CK1-DAG: [[ABEGIN:%.+]] = getelementptr inbounds %class.C, %class.C* [[OBJ]], i32 0, i32 0
// CK1-DAG: [[PRESIZE:%.+]] = call i64 @__tgt_mapper_num_components(i8* [[HANDLE]])
// CK1-DAG: [[SHIPRESIZE:%.+]] = shl i64 [[PRESIZE]], 48
// CK1-DAG: [[BPTRADDR1BC:%.+]] = bitcast %class.C* [[OBJ]] to i8*
// CK1-DAG: [[PTRADDR1BC:%.+]] = bitcast i32* [[ABEGIN]] to i8*
// CK1-DAG: br label %[[MEMBER:[^,]+]]
// CK1-DAG: [[MEMBER]]
// CK1-DAG: br i1 true, label %[[LTYPE:[^,]+]], label %[[MEMBERCOM:[^,]+]]
// CK1-DAG: [[MEMBERCOM]]
// CK1-DAG: [[MEMBERCOMTYPE:%.+]] = add nuw i64 35, [[SHIPRESIZE]]
// CK1-DAG: br label %[[LTYPE]]
// CK1-DAG: [[LTYPE]]
// CK1-DAG: [[MEMBERTYPE:%.+]] = phi i64 [ 35, %[[MEMBER]] ], [ [[MEMBERCOMTYPE]], %[[MEMBERCOM]] ]
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
// CK1: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTRADDR1BC]], i8* [[PTRADDR1BC]], i64 4, i64 [[TYPE1]])
// CK1: [[PTRNEXT]] = getelementptr %class.C*, %class.C** [[PTR]], i32 1
// CK1: [[ISDONE:%.+]] = icmp eq %class.C** [[PTRNEXT]], [[PTREND]]
// CK1: br i1 [[ISDONE]], label %[[LEXIT:[^,]+]], label %[[LBODY]]

// CK1: [[LEXIT]]
// CK1: [[ISARRAY:%.+]] = icmp sge i64 [[SIZE]], 1
// CK1: br i1 [[ISARRAY]], label %[[EVALDEL:[^,]+]], label %[[DONE]]
// CK1: [[EVALDEL]]
// CK1: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK1: [[ISDEL:%.+]] = icmp ne i64 [[TYPEDEL]], 0
// CK1: br i1 [[ISDEL]], label %[[DEL:[^,]+]], label %[[DONE]]
// CK1: [[DEL]]
// CK1-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 4
// CK1-DAG: [[DTYPE:%.+]] = and i64 [[TYPE]], -4
// CK1: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTR]], i8* [[BEGIN]], i64 [[ARRSIZE]], i64 [[DTYPE]])
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

// CK2: define {{.*}}void [[BMPRFUNC:@[.]omp_mapper[.].*B[.]default]](i8*{{.*}}, i8*{{.*}}, i8*{{.*}}, i64{{.*}}, i64{{.*}})

// CK2-LABEL: define {{.*}}void @.omp_mapper.{{.*}}C{{.*}}.id(i8*{{.*}}, i8*{{.*}}, i8*{{.*}}, i64{{.*}}, i64{{.*}})
// CK2: store i8* %{{[^,]+}}, i8** [[HANDLEADDR:%[^,]+]]
// CK2: store i8* %{{[^,]+}}, i8** [[BPTRADDR:%[^,]+]]
// CK2: store i8* %{{[^,]+}}, i8** [[VPTRADDR:%[^,]+]]
// CK2: store i64 %{{[^,]+}}, i{{64|32}}* [[SIZEADDR:%[^,]+]]
// CK2: store i64 %{{[^,]+}}, i64* [[TYPEADDR:%[^,]+]]
// CK2-DAG: [[BYTESIZE:%.+]] = load i64, i64* [[SIZEADDR]]
// CK2-DAG: [[SIZE:%.+]] = udiv exact i64 [[BYTESIZE]], 16
// CK2-DAG: [[TYPE:%.+]] = load i64, i64* [[TYPEADDR]]
// CK2-DAG: [[HANDLE:%.+]] = load i8*, i8** [[HANDLEADDR]]
// CK2-DAG: [[PTRBEGIN:%.+]] = bitcast i8** [[VPTRADDR]] to %class.C**
// CK2-DAG: [[PTREND:%.+]] = getelementptr %class.C*, %class.C** [[PTRBEGIN]], i64 [[SIZE]]
// CK2-DAG: [[BPTR:%.+]] = load i8*, i8** [[BPTRADDR]]
// CK2-DAG: [[BEGIN:%.+]] = load i8*, i8** [[VPTRADDR]]
// CK2: [[ISARRAY:%.+]] = icmp sge i64 [[SIZE]], 1
// CK2: br i1 [[ISARRAY]], label %[[INITEVALDEL:[^,]+]], label %[[LHEAD:[^,]+]]

// CK2: [[INITEVALDEL]]
// CK2: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK2: [[ISNOTDEL:%.+]] = icmp eq i64 [[TYPEDEL]], 0
// CK2: br i1 [[ISNOTDEL]], label %[[INIT:[^,]+]], label %[[LHEAD:[^,]+]]
// CK2: [[INIT]]
// CK2-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 16
// CK2-DAG: [[ITYPE:%.+]] = and i64 [[TYPE]], -4
// CK2: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTR]], i8* [[BEGIN]], i64 [[ARRSIZE]], i64 [[ITYPE]])
// CK2: br label %[[LHEAD:[^,]+]]

// CK2: [[LHEAD]]
// CK2: [[ISEMPTY:%.+]] = icmp eq %class.C** [[PTRBEGIN]], [[PTREND]]
// CK2: br i1 [[ISEMPTY]], label %[[DONE:[^,]+]], label %[[LBODY:[^,]+]]
// CK2: [[LBODY]]
// CK2: [[PTR:%.+]] = phi %class.C** [ [[PTRBEGIN]], %[[LHEAD]] ], [ [[PTRNEXT:%.+]], %[[LCORRECT:[^,]+]] ]
// CK2: [[OBJ:%.+]] = load %class.C*, %class.C** [[PTR]]
// CK2-DAG: [[BBEGIN:%.+]] = getelementptr inbounds %class.C, %class.C* [[OBJ]], i32 0, i32 1
// CK2-DAG: [[PRESIZE:%.+]] = call i64 @__tgt_mapper_num_components(i8* [[HANDLE]])
// CK2-DAG: [[SHIPRESIZE:%.+]] = shl i64 [[PRESIZE]], 48
// CK2-DAG: [[BPTRADDR1BC:%.+]] = bitcast %class.C* [[OBJ]] to i8*
// CK2-DAG: [[PTRADDR1BC:%.+]] = bitcast %class.B* [[BBEGIN]] to i8*
// CK2-DAG: br label %[[MEMBER:[^,]+]]
// CK2-DAG: [[MEMBER]]
// CK2-DAG: br i1 true, label %[[LTYPE:[^,]+]], label %[[MEMBERCOM:[^,]+]]
// CK2-DAG: [[MEMBERCOM]]
// CK2-DAG: [[MEMBERCOMTYPE:%.+]] = add nuw i64 35, [[SHIPRESIZE]]
// CK2-DAG: br label %[[LTYPE]]
// CK2-DAG: [[LTYPE]]
// CK2-DAG: [[MEMBERTYPE:%.+]] = phi i64 [ 35, %[[MEMBER]] ], [ [[MEMBERCOMTYPE]], %[[MEMBERCOM]] ]
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
// CK2: call void [[BMPRFUNC]](i8* [[HANDLE]], i8* [[BPTRADDR1BC]], i8* [[PTRADDR1BC]], i64 8, i64 [[TYPE1]])
// CK2: [[PTRNEXT]] = getelementptr %class.C*, %class.C** [[PTR]], i32 1
// CK2: [[ISDONE:%.+]] = icmp eq %class.C** [[PTRNEXT]], [[PTREND]]
// CK2: br i1 [[ISDONE]], label %[[LEXIT:[^,]+]], label %[[LBODY]]

// CK2: [[LEXIT]]
// CK2: [[ISARRAY:%.+]] = icmp sge i64 [[SIZE]], 1
// CK2: br i1 [[ISARRAY]], label %[[EVALDEL:[^,]+]], label %[[DONE]]
// CK2: [[EVALDEL]]
// CK2: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK2: [[ISDEL:%.+]] = icmp ne i64 [[TYPEDEL]], 0
// CK2: br i1 [[ISDEL]], label %[[DEL:[^,]+]], label %[[DONE]]
// CK2: [[DEL]]
// CK2-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 16
// CK2-DAG: [[DTYPE:%.+]] = and i64 [[TYPE]], -4
// CK2: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTR]], i8* [[BEGIN]], i64 [[ARRSIZE]], i64 [[DTYPE]])
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

// CK3: define {{.*}}void [[MPRFUNC:@[.]omp_mapper[.].*C[.]id]](i8*{{.*}}, i8*{{.*}}, i8*{{.*}}, i64{{.*}}, i64{{.*}})

// CK3-LABEL: define {{.*}}void @{{.*}}foo{{.*}}
void foo(int a){
  // CK3-DAG: [[CVAL:%.+]] = alloca [10 x %class.C]
  // CK3-DAG: [[BVAL:%.+]] = alloca %class.B
  C c[10];
  B b;

  // CK3-DAG: [[BC:%.+]] = getelementptr inbounds %class.B, %class.B* [[BVAL]], i32 0, i32 0

  // CK3-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 2, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null, i8** [[MPRGEP:%.+]])
  // CK3-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK3-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK3-DAG: [[MPRGEP]] = bitcast [2 x i8*]* [[MPR:%[^,]+]] to i8**
  // CK3-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK3-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK3-DAG: [[MPR1:%.+]] = getelementptr inbounds {{.+}}[[MPR]], i{{64|32}} 0, i{{64|32}} 0
  // CK3-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to %class.B**
  // CK3-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to %class.C**
  // CK3-DAG: store %class.B* [[BVAL]], %class.B** [[CBP1]]
  // CK3-DAG: store %class.C* [[BC]], %class.C** [[CP1]]
  // CK3-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[MPR1]]
  // CK3-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK3-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK3-DAG: [[MPR2:%.+]] = getelementptr inbounds {{.+}}[[MPR]], i{{64|32}} 0, i{{64|32}} 1
  // CK3-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [10 x %class.C]**
  // CK3-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to %class.C**
  // CK3-DAG: store [10 x %class.C]* [[CVAL]], [10 x %class.C]** [[CBP2]]
  // CK3-DAG: [[CVALGEP:%.+]] = getelementptr inbounds {{.+}}[[CVAL]], i{{64|32}} 0, i{{64|32}} 0
  // CK3-DAG: store %class.C* [[CVALGEP]], %class.C** [[CP2]]
  // CK3-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[MPR2]]
  // CK3: call void [[KERNEL:@.+]](%class.B* [[BVAL]], [10 x %class.C]* [[CVAL]])
  #pragma omp target map(mapper(id),tofrom: c[0:10], b.c)
  for (int i = 0; i < 10; i++) {
    b.c.a += ++c[i].a;
  }
}


// CK3: define internal void [[KERNEL]](%class.B* {{[^,]+}}, [10 x %class.C]* {{[^,]+}})

#endif // CK3

///==========================================================================///
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix CK4 --check-prefix CK4-64 %s
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix CK4 --check-prefix CK4-64 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix CK4 --check-prefix CK4-32 %s
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix CK4 --check-prefix CK4-32 %s

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm -femit-all-decls -disable-llvm-passes %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -femit-all-decls -disable-llvm-passes -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -femit-all-decls -disable-llvm-passes -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s

#ifdef CK4
// Mapper function code generation and runtime interface.

// CK4-64: [[TSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK4-32: [[TSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// PRESENT=0x1000 | TARGET_PARAM=0x20 | TO=0x1 = 0x1021
// CK4: [[TTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1021]]]

// CK4-64: [[FSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK4-32: [[FSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// PRESENT=0x1000 | TARGET_PARAM=0x20 | FROM=0x2 = 0x1022
// CK4: [[FTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1022]]]

class C {
public:
  int a;
  double *b;
};

#pragma omp declare mapper(id: C s) map(s.a, s.b[0:2])

// CK4: define {{.*}}void [[MPRFUNC:@[.]omp_mapper[.].*C[.]id]](i8*{{.*}}, i8*{{.*}}, i8*{{.*}}, i64{{.*}}, i64{{.*}})
// CK4: store i8* %{{[^,]+}}, i8** [[HANDLEADDR:%[^,]+]]
// CK4: store i8* %{{[^,]+}}, i8** [[BPTRADDR:%[^,]+]]
// CK4: store i8* %{{[^,]+}}, i8** [[VPTRADDR:%[^,]+]]
// CK4: store i64 %{{[^,]+}}, i{{64|32}}* [[SIZEADDR:%[^,]+]]
// CK4: store i64 %{{[^,]+}}, i64* [[TYPEADDR:%[^,]+]]
// CK4-DAG: [[BYTESIZE:%.+]] = load i64, i64* [[SIZEADDR]]
// CK4-64-DAG: [[SIZE:%.+]] = udiv exact i64 [[BYTESIZE]], 16
// CK4-32-DAG: [[SIZE:%.+]] = udiv exact i64 [[BYTESIZE]], 8
// CK4-DAG: [[TYPE:%.+]] = load i64, i64* [[TYPEADDR]]
// CK4-DAG: [[HANDLE:%.+]] = load i8*, i8** [[HANDLEADDR]]
// CK4-DAG: [[PTRBEGIN:%.+]] = bitcast i8** [[VPTRADDR]] to %class.C**
// CK4-DAG: [[PTREND:%.+]] = getelementptr %class.C*, %class.C** [[PTRBEGIN]], i64 [[SIZE]]
// CK4-DAG: [[BPTR:%.+]] = load i8*, i8** [[BPTRADDR]]
// CK4-DAG: [[BEGIN:%.+]] = load i8*, i8** [[VPTRADDR]]
// CK4: [[ISARRAY:%.+]] = icmp sge i64 [[SIZE]], 1
// CK4: br i1 [[ISARRAY]], label %[[INITEVALDEL:[^,]+]], label %[[LHEAD:[^,]+]]

// CK4: [[INITEVALDEL]]
// CK4: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK4: [[ISNOTDEL:%.+]] = icmp eq i64 [[TYPEDEL]], 0
// CK4: br i1 [[ISNOTDEL]], label %[[INIT:[^,]+]], label %[[LHEAD:[^,]+]]
// CK4: [[INIT]]
// CK4-64-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 16
// CK4-32-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 8
// CK4-DAG: [[ITYPE:%.+]] = and i64 [[TYPE]], -4
// CK4: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTR]], i8* [[BEGIN]], i64 [[ARRSIZE]], i64 [[ITYPE]])
// CK4: br label %[[LHEAD:[^,]+]]

// CK4: [[LHEAD]]
// CK4: [[ISEMPTY:%.+]] = icmp eq %class.C** [[PTRBEGIN]], [[PTREND]]
// CK4: br i1 [[ISEMPTY]], label %[[DONE:[^,]+]], label %[[LBODY:[^,]+]]
// CK4: [[LBODY]]
// CK4: [[PTR:%.+]] = phi %class.C** [ [[PTRBEGIN]], %[[LHEAD]] ], [ [[PTRNEXT:%.+]], %[[LCORRECT:[^,]+]] ]
// CK4: [[OBJ:%.+]] = load %class.C*, %class.C** [[PTR]]
// CK4-DAG: [[ABEGIN:%.+]] = getelementptr inbounds %class.C, %class.C* [[OBJ]], i32 0, i32 0
// CK4-DAG: [[BBEGIN:%.+]] = getelementptr inbounds %class.C, %class.C* [[OBJ]], i32 0, i32 1
// CK4-DAG: [[BBEGIN2:%.+]] = getelementptr inbounds %class.C, %class.C* [[OBJ]], i32 0, i32 1
// CK4-DAG: [[BARRBEGIN:%.+]] = load double*, double** [[BBEGIN2]]
// CK4-DAG: [[BARRBEGINGEP:%.+]] = getelementptr inbounds double, double* [[BARRBEGIN]], i[[sz:64|32]] 0
// CK4-DAG: [[BEND:%.+]] = getelementptr double*, double** [[BBEGIN]], i32 1
// CK4-DAG: [[ABEGINV:%.+]] = bitcast i32* [[ABEGIN]] to i8*
// CK4-DAG: [[BENDV:%.+]] = bitcast double** [[BEND]] to i8*
// CK4-DAG: [[ABEGINI:%.+]] = ptrtoint i8* [[ABEGINV]] to i64
// CK4-DAG: [[BENDI:%.+]] = ptrtoint i8* [[BENDV]] to i64
// CK4-DAG: [[CSIZE:%.+]] = sub i64 [[BENDI]], [[ABEGINI]]
// CK4-DAG: [[CUSIZE:%.+]] = sdiv exact i64 [[CSIZE]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
// CK4-DAG: [[BPTRADDR0BC:%.+]] = bitcast %class.C* [[OBJ]] to i8*
// CK4-DAG: [[PTRADDR0BC:%.+]] = bitcast i32* [[ABEGIN]] to i8*
// CK4-DAG: [[PRESIZE:%.+]] = call i64 @__tgt_mapper_num_components(i8* [[HANDLE]])
// CK4-DAG: [[SHIPRESIZE:%.+]] = shl i64 [[PRESIZE]], 48
// CK4-DAG: br label %[[MEMBER:[^,]+]]
// CK4-DAG: [[MEMBER]]
// CK4-DAG: br i1 true, label %[[LTYPE:[^,]+]], label %[[MEMBERCOM:[^,]+]]
// CK4-DAG: [[MEMBERCOM]]
// CK4-DAG: [[MEMBERCOMTYPE:%.+]] = add nuw i64 32, [[SHIPRESIZE]]
// CK4-DAG: br label %[[LTYPE]]
// CK4-DAG: [[LTYPE]]
// CK4-DAG: [[MEMBERTYPE:%.+]] = phi i64 [ 32, %[[MEMBER]] ], [ [[MEMBERCOMTYPE]], %[[MEMBERCOM]] ]
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
// CK4: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTRADDR0BC]], i8* [[PTRADDR0BC]], i64 [[CUSIZE]], i64 [[PHITYPE0]])
// CK4-DAG: [[BPTRADDR1BC:%.+]] = bitcast %class.C* [[OBJ]] to i8*
// CK4-DAG: [[PTRADDR1BC:%.+]] = bitcast i32* [[ABEGIN]] to i8*
// CK4-DAG: br label %[[MEMBER:[^,]+]]
// CK4-DAG: [[MEMBER]]
// CK4-DAG: br i1 false, label %[[LTYPE:[^,]+]], label %[[MEMBERCOM:[^,]+]]
// CK4-DAG: [[MEMBERCOM]]
// 281474976710659 == 0x1,000,000,003
// CK4-DAG: [[MEMBERCOMTYPE:%.+]] = add nuw i64 281474976710659, [[SHIPRESIZE]]
// CK4-DAG: br label %[[LTYPE]]
// CK4-DAG: [[LTYPE]]
// CK4-DAG: [[MEMBERTYPE:%.+]] = phi i64 [ 281474976710659, %[[MEMBER]] ], [ [[MEMBERCOMTYPE]], %[[MEMBERCOM]] ]
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
// CK4: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTRADDR1BC]], i8* [[PTRADDR1BC]], i64 4, i64 [[TYPE1]])
// CK4-DAG: [[BPTRADDR2BC:%.+]] = bitcast double** [[BBEGIN]] to i8*
// CK4-DAG: [[PTRADDR2BC:%.+]] = bitcast double* [[BARRBEGINGEP]] to i8*
// CK4-DAG: br label %[[MEMBER:[^,]+]]
// CK4-DAG: [[MEMBER]]
// CK4-DAG: br i1 false, label %[[LTYPE:[^,]+]], label %[[MEMBERCOM:[^,]+]]
// CK4-DAG: [[MEMBERCOM]]
// 281474976710675 == 0x1,000,000,013
// CK4-DAG: [[MEMBERCOMTYPE:%.+]] = add nuw i64 281474976710675, [[SHIPRESIZE]]
// CK4-DAG: br label %[[LTYPE]]
// CK4-DAG: [[LTYPE]]
// CK4-DAG: [[MEMBERTYPE:%.+]] = phi i64 [ 281474976710675, %[[MEMBER]] ], [ [[MEMBERCOMTYPE]], %[[MEMBERCOM]] ]
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
// CK4: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTRADDR2BC]], i8* [[PTRADDR2BC]], i64 16, i64 [[TYPE2]])
// CK4: [[PTRNEXT]] = getelementptr %class.C*, %class.C** [[PTR]], i32 1
// CK4: [[ISDONE:%.+]] = icmp eq %class.C** [[PTRNEXT]], [[PTREND]]
// CK4: br i1 [[ISDONE]], label %[[LEXIT:[^,]+]], label %[[LBODY]]

// CK4: [[LEXIT]]
// CK4: [[ISARRAY:%.+]] = icmp sge i64 [[SIZE]], 1
// CK4: br i1 [[ISARRAY]], label %[[EVALDEL:[^,]+]], label %[[DONE]]
// CK4: [[EVALDEL]]
// CK4: [[TYPEDEL:%.+]] = and i64 [[TYPE]], 8
// CK4: [[ISDEL:%.+]] = icmp ne i64 [[TYPEDEL]], 0
// CK4: br i1 [[ISDEL]], label %[[DEL:[^,]+]], label %[[DONE]]
// CK4: [[DEL]]
// CK4-64-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 16
// CK4-32-DAG: [[ARRSIZE:%.+]] = mul nuw i64 [[SIZE]], 8
// CK4-DAG: [[DTYPE:%.+]] = and i64 [[TYPE]], -4
// CK4: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTR]], i8* [[BEGIN]], i64 [[ARRSIZE]], i64 [[DTYPE]])
// CK4: br label %[[DONE]]
// CK4: [[DONE]]
// CK4: ret void


// CK4-LABEL: define {{.*}}void @{{.*}}foo{{.*}}
void foo(int a){
  int i = a;
  C c;
  c.a = a;

  // CK4-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[TGEPBP:%.+]], i8** [[TGEPP:%.+]], i64* getelementptr {{.+}}[1 x i64]* [[TSIZES]], i32 0, i32 0), {{.+}}getelementptr {{.+}}[1 x i64]* [[TTYPES]]{{.+}}, i8** null, i8** [[TMPRGEP:%.+]])
  // CK4-DAG: [[TGEPBP]] = getelementptr inbounds {{.+}}[[TBP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[TGEPP]] = getelementptr inbounds {{.+}}[[TP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[TMPRGEP]] = bitcast [1 x i8*]* [[TMPR:%[^,]+]] to i8**
  // CK4-DAG: [[TBP0:%.+]] = getelementptr inbounds {{.+}}[[TBP]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[TP0:%.+]] = getelementptr inbounds {{.+}}[[TP]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[TMPR1:%.+]] = getelementptr inbounds {{.+}}[[TMPR]], i[[sz]] 0, i[[sz]] 0
  // CK4-DAG: [[TCBP0:%.+]] = bitcast i8** [[TBP0]] to %class.C**
  // CK4-DAG: [[TCP0:%.+]] = bitcast i8** [[TP0]] to %class.C**
  // CK4-DAG: store %class.C* [[VAL:%[^,]+]], %class.C** [[TCBP0]]
  // CK4-DAG: store %class.C* [[VAL]], %class.C** [[TCP0]]
  // CK4-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[TMPR1]]
  #pragma omp target update to(present, mapper(id): c)

  // CK4-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[FGEPBP:%.+]], i8** [[FGEPP:%.+]], i64* getelementptr {{.+}}[1 x i64]* [[FSIZES]], i32 0, i32 0), {{.+}}getelementptr {{.+}}[1 x i64]* [[FTYPES]]{{.+}}, i8** null, i8** [[FMPRGEP:%.+]])
  // CK4-DAG: [[FGEPBP]] = getelementptr inbounds {{.+}}[[FBP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[FGEPP]] = getelementptr inbounds {{.+}}[[FP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[FMPRGEP]] = bitcast [1 x i8*]* [[FMPR:%[^,]+]] to i8**
  // CK4-DAG: [[FBP0:%.+]] = getelementptr inbounds {{.+}}[[FBP]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[FP0:%.+]] = getelementptr inbounds {{.+}}[[FP]], i{{.+}} 0, i{{.+}} 0
  // CK4-DAG: [[FMPR1:%.+]] = getelementptr inbounds {{.+}}[[FMPR]], i[[sz]] 0, i[[sz]] 0
  // CK4-DAG: [[FCBP0:%.+]] = bitcast i8** [[FBP0]] to %class.C**
  // CK4-DAG: [[FCP0:%.+]] = bitcast i8** [[FP0]] to %class.C**
  // CK4-DAG: store %class.C* [[VAL]], %class.C** [[FCBP0]]
  // CK4-DAG: store %class.C* [[VAL]], %class.C** [[FCP0]]
  // CK4-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64)* [[MPRFUNC]] to i8*), i8** [[FMPR1]]
  #pragma omp target update from(mapper(id), present: c)
}

#endif // CK4

#endif // HEADER
