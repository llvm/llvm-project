// RUN: %clang_cc1 -std=c++11 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -triple i386-unknown-unknown %s -emit-llvm -fsanitize=signed-integer-overflow -o - | FileCheck --check-prefix=SIO %s

// CHECK: @[[ABC4:.*]] = {{.*}} constant [4 x i8] c"abc\00"
// CHECK: @[[ABC15:.*]] = {{.*}} constant [15 x i8] c"abc\00\00\00\00

// CHECK-LABEL: define{{.*}} void @_Z2fni
void fn(int n) {
  // CHECK: icmp ult i{{32|64}} %{{[^ ]+}}, 3
  // CHECK: store i32 1
  // CHECK: store i32 2
  // CHECK: store i32 3
  // CHECK: sub {{.*}}, 12
  // CHECK: call void @llvm.memset
  new int[n] { 1, 2, 3 };
}

// CHECK-LABEL: define{{.*}} void @_Z11const_exactv
void const_exact() {
  // CHECK-NOT: icmp ult i{{32|64}} %{{[^ ]+}}, 3
  // CHECK-NOT: icmp eq i32*
  new int[3] { 1, 2, 3 };
}

// CHECK-LABEL: define{{.*}} void @_Z16const_sufficientv
void const_sufficient() {
  // CHECK-NOT: icmp ult i{{32|64}} %{{[^ ]+}}, 3
  new int[4] { 1, 2, 3 };
  // CHECK: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z22check_array_value_initv
void check_array_value_init() {
  struct S;
  new (int S::*[3][4][5]) ();

  // CHECK: call noalias noundef nonnull i8* @_Zna{{.}}(i{{32 noundef 240|64 noundef 480}})
  // CHECK: getelementptr inbounds i{{32|64}}, i{{32|64}}* {{.*}}, i{{32|64}} 60

  // CHECK: phi
  // CHECK: store i{{32|64}} -1,
  // CHECK: getelementptr inbounds i{{32|64}}, i{{32|64}}* {{.*}}, i{{32|64}} 1
  // CHECK: icmp eq
  // CHECK: br i1
}

// CHECK-LABEL: define{{.*}} void @_Z15string_nonconsti
void string_nonconst(int n) {
  // CHECK: icmp slt i{{32|64}} %{{[^ ]+}}, 4
  // FIXME: Conditionally throw an exception rather than passing -1 to alloc function
  // CHECK: select
  // CHECK: %[[PTR:.*]] = call noalias noundef nonnull i8* @_Zna{{.}}(i{{32|64}}
  // CHECK: call void @llvm.memcpy{{.*}}(i8* align {{[0-9]+}} %[[PTR]], i8* align {{[0-9]+}} getelementptr inbounds ([4 x i8], [4 x i8]* @[[ABC4]], i32 0, i32 0), i32 4,
  // CHECK: %[[REST:.*]] = getelementptr inbounds i8, i8* %[[PTR]], i32 4
  // CHECK: %[[RESTSIZE:.*]] = sub {{.*}}, 4
  // CHECK: call void @llvm.memset{{.*}}(i8* align {{[0-9]+}} %[[REST]], i8 0, i{{32|64}} %[[RESTSIZE]],
  new char[n] { "abc" };
}

// CHECK-LABEL: define{{.*}} void @_Z12string_exactv
void string_exact() {
  // CHECK-NOT: icmp
  // CHECK: %[[PTR:.*]] = call noalias noundef nonnull i8* @_Zna{{.}}(i{{32|64}} noundef 4)
  // CHECK: call void @llvm.memcpy{{.*}}(i8* align {{[0-9]+}} %[[PTR]], i8* align {{[0-9]+}} getelementptr inbounds ([4 x i8], [4 x i8]* @[[ABC4]], i32 0, i32 0), i32 4,
  // CHECK-NOT: memset
  new char[4] { "abc" };
}

// CHECK-LABEL: define{{.*}} void @_Z17string_sufficientv
void string_sufficient() {
  // CHECK-NOT: icmp
  // CHECK: %[[PTR:.*]] = call noalias noundef nonnull i8* @_Zna{{.}}(i{{32|64}} noundef 15)
  // FIXME: For very large arrays, it would be preferable to emit a small copy and a memset.
  // CHECK: call void @llvm.memcpy{{.*}}(i8* align {{[0-9]+}} %[[PTR]], i8* align {{[0-9]+}} getelementptr inbounds ([15 x i8], [15 x i8]* @[[ABC15]], i32 0, i32 0), i32 15,
  // CHECK-NOT: memset
  new char[15] { "abc" };
}

// CHECK-LABEL: define{{.*}} void @_Z10aggr_exactv
void aggr_exact() {
  // CHECK-NOT: icmp
  // CHECK: %[[MEM:.*]] = call noalias noundef nonnull i8* @_Zna{{.}}(i{{32|64}} noundef 16)
  // CHECK: %[[PTR0:.*]] = bitcast i8* %[[MEM]] to %[[AGGR:.*]]*
  // CHECK: %[[FIELD:.*]] = getelementptr inbounds %[[AGGR]], %[[AGGR]]* %[[PTR0]], i32 0, i32 0{{$}}
  // CHECK: store i32 1, i32* %[[FIELD]]
  // CHECK: %[[FIELD:.*]] = getelementptr inbounds %[[AGGR]], %[[AGGR]]* %[[PTR0]], i32 0, i32 1{{$}}
  // CHECK: store i32 2, i32* %[[FIELD]]
  // CHECK: %[[PTR1:.*]] = getelementptr inbounds %[[AGGR]], %[[AGGR]]* %[[PTR0]], i32 1{{$}}
  // CHECK: %[[FIELD:.*]] = getelementptr inbounds %[[AGGR]], %[[AGGR]]* %[[PTR1]], i32 0, i32 0{{$}}
  // CHECK: store i32 3, i32* %[[FIELD]]
  // CHECK: %[[FIELD:.*]] = getelementptr inbounds %[[AGGR]], %[[AGGR]]* %[[PTR1]], i32 0, i32 1{{$}}
  // CHECK: store i32 0, i32* %[[FIELD]]
  // CHECK-NOT: store
  // CHECK-NOT: memset
  struct Aggr { int a, b; };
  new Aggr[2] { 1, 2, 3 };
}

// CHECK-LABEL: define{{.*}} void @_Z15aggr_sufficienti
void aggr_sufficient(int n) {
  // CHECK: icmp ult i32 %{{.*}}, 2
  // CHECK: %[[MEM:.*]] = call noalias noundef nonnull i8* @_Zna{{.}}(
  // CHECK: %[[PTR0:.*]] = bitcast i8* %[[MEM]] to %[[AGGR:.*]]*
  // CHECK: %[[FIELD:.*]] = getelementptr inbounds %[[AGGR]], %[[AGGR]]* %[[PTR0]], i32 0, i32 0{{$}}
  // CHECK: store i32 1, i32* %[[FIELD]]
  // CHECK: %[[FIELD:.*]] = getelementptr inbounds %[[AGGR]], %[[AGGR]]* %[[PTR0]], i32 0, i32 1{{$}}
  // CHECK: store i32 2, i32* %[[FIELD]]
  // CHECK: %[[PTR1:.*]] = getelementptr inbounds %[[AGGR]], %[[AGGR]]* %[[PTR0]], i32 1{{$}}
  // CHECK: %[[FIELD:.*]] = getelementptr inbounds %[[AGGR]], %[[AGGR]]* %[[PTR1]], i32 0, i32 0{{$}}
  // CHECK: store i32 3, i32* %[[FIELD]]
  // CHECK: %[[FIELD:.*]] = getelementptr inbounds %[[AGGR]], %[[AGGR]]* %[[PTR1]], i32 0, i32 1{{$}}
  // CHECK: store i32 0, i32* %[[FIELD]]
  // CHECK: %[[PTR2:.*]] = getelementptr inbounds %[[AGGR]], %[[AGGR]]* %[[PTR1]], i32 1{{$}}
  // CHECK: %[[REMAIN:.*]] = sub i32 {{.*}}, 16
  // CHECK: %[[MEM:.*]] = bitcast %[[AGGR]]* %[[PTR2]] to i8*
  // CHECK: call void @llvm.memset{{.*}}(i8* align {{[0-9]+}} %[[MEM]], i8 0, i32 %[[REMAIN]],
  struct Aggr { int a, b; };
  new Aggr[n] { 1, 2, 3 };
}

// SIO-LABEL: define{{.*}} void @_Z14constexpr_testv
void constexpr_test() {
  // SIO: call noalias noundef nonnull i8* @_Zna{{.}}(i32 noundef 4)
  new int[0+1]{0};
}

// CHECK-LABEL: define{{.*}} void @_Z13unknown_boundv
void unknown_bound() {
  struct Aggr { int x, y, z; };
  new Aggr[]{1, 2, 3, 4};
  // CHECK: call {{.*}}_Znaj(i32 noundef 24)
  // CHECK: store i32 1
  // CHECK: store i32 2
  // CHECK: store i32 3
  // CHECK: store i32 4
  // CHECK: store i32 0
  // CHECK: store i32 0
  // CHECK-NOT: store
  // CHECK: }
}

// CHECK-LABEL: define{{.*}} void @_Z20unknown_bound_stringv
void unknown_bound_string() {
  new char[]{"hello"};
  // CHECK: call {{.*}}_Znaj(i32 noundef 6)
  // CHECK: memcpy{{.*}} i32 6,
}
