// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s

// CHECK-DAG: [[ONE_STR:@.*]] = private unnamed_addr constant [4 x i8] c"One\00"
// CHECK-DAG: [[ARRAY_STR:@.*]] = private unnamed_addr constant [6 x i8] c"Array\00"
// CHECK-DAG: [[ONEWITHCOUNTER_STR:@.*]] = private unnamed_addr constant [15 x i8] c"OneWithCounter\00"
// CHECK-DAG: [[ARRAYWITHCOUNTER_STR:@.*]] = private unnamed_addr constant [17 x i8] c"ArrayWithCounter\00"
// CHECK-NOT: private unnamed_addr constant [{{[0-9]+}} x i8] c"Static

RWBuffer<float> One : register(u1, space5);
RWBuffer<float> Array[2] : register(u10, space6);
RWStructuredBuffer<int> OneWithCounter : register(u2, space4);
RWStructuredBuffer<int> ArrayWithCounter[2] : register(u7, space4);

// Check that the non-static resource One is initialized from binding on
// startup (register 1, space 5).
// CHECK: define internal void @__cxx_global_var_init{{.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} @One, i32 noundef 1, i32 noundef 5, i32 noundef 1, i32 noundef 0, ptr noundef [[ONE_STR]])

// Check that the non-static resource OneWithCounter is initialized from binding on
// startup (register 2, space 4).
// CHECK: define internal void @__cxx_global_var_init{{.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @hlsl::RWStructuredBuffer<int>::__createFromBindingWithImplicitCounter(unsigned int, unsigned int, int, unsigned int, char const*, unsigned int)
// CHECK-SAME: (ptr {{.*}} @OneWithCounter, i32 noundef 2, i32 noundef 4, i32 noundef 1, i32 noundef 0, ptr noundef [[ONEWITHCOUNTER_STR]], i32 noundef 0)

// Note that non-static resource arrays are not initialized on startup.
// The individual resources from the array are initialized on access.

static RWBuffer<float> StaticOne;
static RWBuffer<float> StaticArray[2];

// Check that StaticOne resource is initialized on startup with the default
// constructor and not from binding. It will initalize the handle to poison.
// CHECK: define internal void @__cxx_global_var_init{{.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::RWBuffer()(ptr {{.*}} @StaticOne)

// Check that StaticArray elements are initialized on startup with the default
// constructor and not from binding. The initializer will loop over the array
// elements and call the default constructor for each one, setting the handle to poison.
// CHECK: define internal void @__cxx_global_var_init{{.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: br label %arrayctor.loop
// CHECK: arrayctor.loop:                                   ; preds = %arrayctor.loop, %entry
// CHECK-NEXT:   %arrayctor.cur = phi ptr [ @StaticArray, %entry ], [ %arrayctor.next, %arrayctor.loop ]
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::RWBuffer()(ptr {{.*}} %arrayctor.cur)
// CHECK-NEXT: %arrayctor.next = getelementptr inbounds %"class.hlsl::RWBuffer", ptr %arrayctor.cur, i32 1
// CHECK-NEXT: %arrayctor.done = icmp eq ptr %arrayctor.next, getelementptr inbounds (%"class.hlsl::RWBuffer", ptr @StaticArray, i32 2)
// CHECK-NEXT: br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop
// CHECK: arrayctor.cont:                                   ; preds = %arrayctor.loop
// CHECK-NEXT: ret void

static RWStructuredBuffer<int> StaticOneWithCounter;

// Check that StaticOneWithCounter resource is initialized on startup with the default
// constructor and not from binding. It will initalize the handle to poison.
// CHECK: define internal void @__cxx_global_var_init{{.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @hlsl::RWStructuredBuffer<int>::RWStructuredBuffer()(ptr {{.*}} @StaticOneWithCounter)

// No other global initialization routines should be present.
// CHECK-NOT: define internal void @__cxx_global_var_init{{.*}}

[numthreads(4,1,1)]
void main() {
// CHECK: define internal void @main()()
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[TMP0:.*]] = alloca %"class.hlsl::RWBuffer"

  static RWBuffer<float> StaticLocal;
// Check that StaticLocal is initialized by default constructor (handle set to poison)
// and not from binding.
// call void @hlsl::RWBuffer<float>::RWBuffer()(ptr {{.*}} @main()::StaticLocal)

  StaticLocal = Array[1];
// A[2][0] is accessed here, so it should be initialized from binding (register 10, space 6, index 1),
// and then assigned to StaticLocal using = operator.
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} %[[TMP0]], i32 noundef 10, i32 noundef 6, i32 noundef 2, i32 noundef 1, ptr noundef [[ARRAY_STR]])
// CHECK-NEXT: call {{.*}} ptr @hlsl::RWBuffer<float>::operator=({{.*}})(ptr {{.*}} @main()::StaticLocal, ptr {{.*}} %[[TMP0]])

  StaticOne = One;
// Operator = call to assign non-static One handle to static StaticOne.
// CHECK-NEXT: call {{.*}} ptr @hlsl::RWBuffer<float>::operator=({{.*}})(ptr {{.*}} @StaticOne, ptr {{.*}} @One)

  StaticArray = Array;
// Check that each elements of StaticArray is initialized from binding (register 10, space 6, indices 0 and 1).
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 @StaticArray, i32 noundef 10, i32 noundef 6, i32 noundef 2, i32 noundef 0, ptr noundef [[ARRAY_STR]])
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 getelementptr ([2 x %"class.hlsl::RWBuffer"], ptr @StaticArray, i32 0, i32 1),
// CHECK-SAME: i32 noundef 10, i32 noundef 6, i32 noundef 2, i32 noundef 1, ptr noundef [[ARRAY_STR]]

  StaticArray[1] = One;
// Operator = call to assign non-static One handle to StaticArray element.
// CHECK-NEXT: call {{.*}} ptr @hlsl::RWBuffer<float>::operator=(hlsl::RWBuffer<float> const&)
// CHECK-SAME: (ptr {{.*}} getelementptr inbounds ([2 x %"class.hlsl::RWBuffer"], ptr @StaticArray, i32 0, i32 1), ptr {{.*}} @One)

  StaticLocal[0] = 123;
// CHECK-NEXT: %[[PTR0:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int)(ptr {{.*}} @main()::StaticLocal, i32 noundef 0)
// CHECK-NEXT: store float 1.230000e+02, ptr %[[PTR0]]

  StaticOne[1] = 456;
// CHECK-NEXT: %[[PTR1:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int)(ptr {{.*}}) @StaticOne, i32 noundef 1)
// CHECK-NEXT: store float 4.560000e+02, ptr %[[PTR1]], align 4

  StaticArray[1][2] = 789;
// CHECK-NEXT: %[[PTR2:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int)
// CHECK-SAME: (ptr {{.*}} getelementptr inbounds ([2 x %"class.hlsl::RWBuffer"], ptr @StaticArray, i32 0, i32 1), i32 noundef 2)
// CHECK-NEXT: store float 7.890000e+02, ptr %[[PTR2]], align 4

  static RWStructuredBuffer<int> StaticLocalWithCounter;
// Check that StaticLocalWithCounter is initialized by default constructor (handle set to poison)
// and not from binding.
// call void @hlsl::RWStructuredBuffer<int>::RWStructuredBuffer()(ptr {{.*}} @main()::StaticLocalWithCounter)

  static RWStructuredBuffer<int> StaticLocalArrayWithCounter[2];

  StaticLocalWithCounter = OneWithCounter;
// Operator = call to assign non-static OneWithCounter handles to StaticLocalWithCounter handles.
// CHECK: call {{.*}} ptr @hlsl::RWStructuredBuffer<int>::operator=(hlsl::RWStructuredBuffer<int> const&)(ptr {{.*}} @main()::StaticLocalWithCounter, ptr {{.*}} @OneWithCounter)

  StaticLocalArrayWithCounter = ArrayWithCounter;
// Check that each elements of StaticLocalArrayWithCounter is initialized from binding
// of ArrayWithCounter (register 7, space 4, indices 0 and 1).
// CHECK: call void @hlsl::RWStructuredBuffer<int>::__createFromBindingWithImplicitCounter(unsigned int, unsigned int, int, unsigned int, char const*, unsigned int)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWStructuredBuffer") align 4 @main()::StaticLocalArrayWithCounter,
// CHECK-SAME: i32 noundef 7, i32 noundef 4, i32 noundef 2, i32 noundef 0, ptr noundef [[ARRAYWITHCOUNTER_STR]], i32 noundef 1)

// CHECK-NEXT: call void @hlsl::RWStructuredBuffer<int>::__createFromBindingWithImplicitCounter(unsigned int, unsigned int, int, unsigned int, char const*, unsigned int)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWStructuredBuffer") align 4 getelementptr ([2 x %"class.hlsl::RWStructuredBuffer"], ptr @main()::StaticLocalArrayWithCounter, i32 0, i32 1),
// CHECK-SAME: i32 noundef 7, i32 noundef 4, i32 noundef 2, i32 noundef 1, ptr noundef [[ARRAYWITHCOUNTER_STR]], i32 noundef 1)
}

// No other binding initialization calls should be present.
// CHECK-NOT: call void @hlsl::RWBuffer<float>::__createFrom{{.*}}Binding{{.*}}
