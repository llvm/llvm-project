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

// CK0-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}.region_id = weak constant i8 0
// CK0-64: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 35]
// CK0-64: [[TSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[TSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[TTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 33]
// CK0-64: [[FSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// CK0-32: [[FSIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// CK0: [[FTYPES:@.+]] = {{.+}}constant [1 x i64] [i64 34]

class C {
public:
  int a;
  double *b;
};

#pragma omp declare mapper(id: C s) map(s.a, s.b[0:2])

// CK0-LABEL: define {{.*}}void @.omp_mapper.{{.*}}C.id{{.*}}(i8*{{.*}}, i8*{{.*}}, i8*{{.*}}, i64{{.*}}, i64{{.*}})
// CK0: store i8* %{{[^,]+}}, i8** [[HANDLEADDR:%[^,]+]]
// CK0: store i8* %{{[^,]+}}, i8** [[BPTRADDR:%[^,]+]]
// CK0: store i8* %{{[^,]+}}, i8** [[VPTRADDR:%[^,]+]]
// CK0: store i64 %{{[^,]+}}, i{{64|32}}* [[SIZEADDR:%[^,]+]]
// CK0: store i64 %{{[^,]+}}, i64* [[TYPEADDR:%[^,]+]]
// CK0-DAG: [[SIZE:%.+]] = load i64, i64* [[SIZEADDR]]
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

  // CK0-DAG: call i32 @__tgt_target(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK0-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK0-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK0-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK0-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to %class.C**
  // CK0-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to %class.C**
  // CK0-DAG: store %class.C* [[VAL:%[^,]+]], %class.C** [[CBP1]]
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[CP1]]
  // CK0: call void [[KERNEL:@.+]](%class.C* [[VAL]])
  #pragma omp target map(mapper(id),tofrom: c)
  {
   ++c.a;
  }

  // CK0-DAG: call void @__tgt_target_data_update(i64 -1, i32 1, i8** [[TGEPBP:%.+]], i8** [[TGEPP:%.+]], i64* getelementptr {{.+}}[1 x i64]* [[TSIZES]], i32 0, i32 0), {{.+}}getelementptr {{.+}}[1 x i64]* [[TTYPES]]{{.+}})
  // CK0-DAG: [[TGEPBP]] = getelementptr inbounds {{.+}}[[TBP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[TGEPP]] = getelementptr inbounds {{.+}}[[TP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[TBP0:%.+]] = getelementptr inbounds {{.+}}[[TBP]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[TP0:%.+]] = getelementptr inbounds {{.+}}[[TP]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[TCBP0:%.+]] = bitcast i8** [[TBP0]] to %class.C**
  // CK0-DAG: [[TCP0:%.+]] = bitcast i8** [[TP0]] to %class.C**
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[TCBP0]]
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[TCP0]]
  #pragma omp target update to(mapper(id): c)

  // CK0-DAG: call void @__tgt_target_data_update(i64 -1, i32 1, i8** [[FGEPBP:%.+]], i8** [[FGEPP:%.+]], i64* getelementptr {{.+}}[1 x i64]* [[FSIZES]], i32 0, i32 0), {{.+}}getelementptr {{.+}}[1 x i64]* [[FTYPES]]{{.+}})
  // CK0-DAG: [[FGEPBP]] = getelementptr inbounds {{.+}}[[FBP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[FGEPP]] = getelementptr inbounds {{.+}}[[FP:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[FBP0:%.+]] = getelementptr inbounds {{.+}}[[FBP]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[FP0:%.+]] = getelementptr inbounds {{.+}}[[FP]], i{{.+}} 0, i{{.+}} 0
  // CK0-DAG: [[FCBP0:%.+]] = bitcast i8** [[FBP0]] to %class.C**
  // CK0-DAG: [[FCP0:%.+]] = bitcast i8** [[FP0]] to %class.C**
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[FCBP0]]
  // CK0-DAG: store %class.C* [[VAL]], %class.C** [[FCP0]]
  #pragma omp target update from(mapper(id): c)
}


// CK0: define internal void [[KERNEL]](%class.C* {{.+}}[[ARG:%.+]])
// CK0: [[ADDR:%.+]] = alloca %class.C*,
// CK0: store %class.C* [[ARG]], %class.C** [[ADDR]]
// CK0: [[CADDR:%.+]] = load %class.C*, %class.C** [[ADDR]]
// CK0: [[CAADDR:%.+]] = getelementptr inbounds %class.C, %class.C* [[CADDR]], i32 0, i32 0
// CK0: [[VAL:%[^,]+]] = load i32, i32* [[CAADDR]]
// CK0: {{.+}} = add nsw i32 [[VAL]], 1
// CK0: }

#endif


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
// CK1-DAG: [[SIZE:%.+]] = load i64, i64* [[SIZEADDR]]
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
// CK1-DAG: [[AEND:%.+]] = getelementptr i32, i32* [[ABEGIN]], i32 1
// CK1-DAG: [[ABEGINV:%.+]] = bitcast i32* [[ABEGIN]] to i8*
// CK1-DAG: [[AENDV:%.+]] = bitcast i32* [[AEND]] to i8*
// CK1-DAG: [[ABEGINI:%.+]] = ptrtoint i8* [[ABEGINV]] to i64
// CK1-DAG: [[AENDI:%.+]] = ptrtoint i8* [[AENDV]] to i64
// CK1-DAG: [[CSIZE:%.+]] = sub i64 [[AENDI]], [[ABEGINI]]
// CK1-DAG: [[CUSIZE:%.+]] = sdiv exact i64 [[CSIZE]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
// CK1-DAG: [[BPTRADDR0BC:%.+]] = bitcast %class.C* [[OBJ]] to i8*
// CK1-DAG: [[PTRADDR0BC:%.+]] = bitcast i32* [[ABEGIN]] to i8*
// CK1-DAG: [[PRESIZE:%.+]] = call i64 @__tgt_mapper_num_components(i8* [[HANDLE]])
// CK1-DAG: [[SHIPRESIZE:%.+]] = shl i64 [[PRESIZE]], 48
// CK1-DAG: br label %[[MEMBER:[^,]+]]
// CK1-DAG: [[MEMBER]]
// CK1-DAG: br i1 true, label %[[LTYPE:[^,]+]], label %[[MEMBERCOM:[^,]+]]
// CK1-DAG: [[MEMBERCOM]]
// CK1-DAG: [[MEMBERCOMTYPE:%.+]] = add nuw i64 32, [[SHIPRESIZE]]
// CK1-DAG: br label %[[LTYPE]]
// CK1-DAG: [[LTYPE]]
// CK1-DAG: [[MEMBERTYPE:%.+]] = phi i64 [ 32, %[[MEMBER]] ], [ [[MEMBERCOMTYPE]], %[[MEMBERCOM]] ]
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
// CK1-DAG: [[TYPE0:%.+]] = phi i64 [ [[ALLOCTYPE]], %[[ALLOC]] ], [ [[TOTYPE]], %[[TO]] ], [ [[FROMTYPE]], %[[FROM]] ], [ [[MEMBERTYPE]], %[[TOELSE]] ]
// CK1-64: call void @__tgt_push_mapper_component(i8* [[HANDLE]], i8* [[BPTRADDR0BC]], i8* [[PTRADDR0BC]], i64 [[CUSIZE]], i64 [[TYPE0]])
// CK1-DAG: [[BPTRADDR1BC:%.+]] = bitcast %class.C* [[OBJ]] to i8*
// CK1-DAG: [[PTRADDR1BC:%.+]] = bitcast i32* [[ABEGIN]] to i8*
// CK1-DAG: br label %[[MEMBER:[^,]+]]
// CK1-DAG: [[MEMBER]]
// CK1-DAG: br i1 false, label %[[LTYPE:[^,]+]], label %[[MEMBERCOM:[^,]+]]
// CK1-DAG: [[MEMBERCOM]]
// 281474976710659 == 0x1,000,000,003
// CK1-DAG: [[MEMBERCOMTYPE:%.+]] = add nuw i64 281474976710659, [[SHIPRESIZE]]
// CK1-DAG: br label %[[LTYPE]]
// CK1-DAG: [[LTYPE]]
// CK1-DAG: [[MEMBERTYPE:%.+]] = phi i64 [ 281474976710659, %[[MEMBER]] ], [ [[MEMBERCOMTYPE]], %[[MEMBERCOM]] ]
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

#endif

#endif
