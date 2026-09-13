// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library -ast-dump %s | FileCheck %s

// Global variables for ResourceDescriptorHeap and SamplerDescriptorHeap
// CHECK: VarDecl {{.*}} used ResourceDescriptorHeap 'hlsl_private __hlsl_resource_descriptor_heap_struct' static internal-linkage
// CHECK-NEXT: AvailabilityAttr {{.*}} shadermodel 6.6 0 0 "" "" 0

// CHECK: VarDecl {{.*}} used SamplerDescriptorHeap 'hlsl_private __hlsl_sampler_descriptor_heap_struct' static internal-linkage
// CHECK-NEXT: AvailabilityAttr {{.*}} shadermodel 6.6 0 0 "" "" 0

void useBuffer(RWBuffer<int> Buffer) {}

// CHECK-LABEL: FunctionDecl {{.*}} testInvocations
export void testInvocations(unsigned Index) {

// Buf1 declaration with direct initialization
// CHECK: VarDecl {{.*}} Buf1 'RWBuffer<int>':'hlsl::RWBuffer<int>' cinit
// CHECK-NEXT: ExprWithCleanups{{.*}} 'RWBuffer<int>':'hlsl::RWBuffer<int>'

// RWBuffer copy constructor
// CHECK-NEXT: CXXConstructExpr {{.*}} 'RWBuffer<int>':'hlsl::RWBuffer<int>' 'void (const hlsl::RWBuffer<int> &)'
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'const RWBuffer<int>':'const hlsl::RWBuffer<int>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const RWBuffer<int>':'const hlsl::RWBuffer<int>' <NoOp>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'RWBuffer<int>':'hlsl::RWBuffer<int>' <ConstructorConversion>

// RWBuffer heap info constructor
// CHECK-NEXT: CXXConstructExpr {{.*}} 'RWBuffer<int>':'hlsl::RWBuffer<int>' 'void (hlsl::__hlsl_heap_resource_info)'

// __hlsl_heap_resource_info copy constructor
// CHECK-NEXT: CXXConstructExpr {{.*}} 'hlsl::__hlsl_heap_resource_info' 'void (__hlsl_heap_resource_info &&) noexcept' elidable

// Indexing into ResourceDescriptorHeap
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} '__hlsl_heap_resource_info' xvalue
// CHECK-NEXT: CXXOperatorCallExpr {{.*}} '__hlsl_heap_resource_info' '[]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__hlsl_heap_resource_info (*)(uint32_t)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} '__hlsl_heap_resource_info (uint32_t)' lvalue CXXMethod {{.*}} 'operator[]' '__hlsl_heap_resource_info (uint32_t)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'hlsl::__hlsl_resource_descriptor_heap_struct' lvalue <AddressSpaceConversion>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl_private __hlsl_resource_descriptor_heap_struct' lvalue Var {{.*}} 'ResourceDescriptorHeap' 'hlsl_private __hlsl_resource_descriptor_heap_struct'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'

  RWBuffer<int> Buf1 = ResourceDescriptorHeap[Index];

// Buf2 declaration initialized with default constructor (handle is poison)
// CHECK: VarDecl {{.*}} Buf2 'RWBuffer<int>':'hlsl::RWBuffer<int>' callinit
// CHECK-NEXT: CXXConstructExpr {{.*}} 'RWBuffer<int>':'hlsl::RWBuffer<int>' 'void ()'

// Buf2 assignment operator
// CHECK-NEXT: ExprWithCleanups
// CHECK-NEXT: CXXOperatorCallExpr {{.*}} 'hlsl::RWBuffer<int>' lvalue '='
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'hlsl::RWBuffer<int> &(*)(const hlsl::RWBuffer<int> &)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::RWBuffer<int> &(const hlsl::RWBuffer<int> &)' lvalue CXXMethod {{.*}} 'operator=' 'hlsl::RWBuffer<int> &(const hlsl::RWBuffer<int> &)'
// CHECK-NEXT: DeclRefExpr {{.*}} 'RWBuffer<int>':'hlsl::RWBuffer<int>' lvalue Var {{.*}} 'Buf2' 'RWBuffer<int>':'hlsl::RWBuffer<int>'

// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'const hlsl::RWBuffer<int>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const hlsl::RWBuffer<int>' <NoOp>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'hlsl::RWBuffer<int>' <ConstructorConversion>

// RWBuffer heap info constructor
// CHECK-NEXT: CXXConstructExpr {{.*}} 'hlsl::RWBuffer<int>' 'void (hlsl::__hlsl_heap_resource_info)'

// __hlsl_heap_resource_info copy constructor
// CHECK-NEXT: CXXConstructExpr {{.*}} 'hlsl::__hlsl_heap_resource_info' 'void (__hlsl_heap_resource_info &&) noexcept' elidable

// Indexing into ResourceDescriptorHeap
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} '__hlsl_heap_resource_info' xvalue
// CHECK-NEXT: CXXOperatorCallExpr {{.*}} '__hlsl_heap_resource_info' '[]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__hlsl_heap_resource_info (*)(uint32_t)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} '__hlsl_heap_resource_info (uint32_t)' lvalue CXXMethod {{.*}} 'operator[]' '__hlsl_heap_resource_info (uint32_t)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'hlsl::__hlsl_resource_descriptor_heap_struct' lvalue <AddressSpaceConversion>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl_private __hlsl_resource_descriptor_heap_struct' lvalue Var {{.*}} 'ResourceDescriptorHeap' 'hlsl_private __hlsl_resource_descriptor_heap_struct'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'

  RWBuffer<int> Buf2;
  Buf2 = ResourceDescriptorHeap[Index];

// Call to useBuffer with a temporary RWBuffer constructed from ResourceDescriptorHeap[Index]
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(RWBuffer<int>)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (RWBuffer<int>)' lvalue Function {{.*}} 'useBuffer' 'void (RWBuffer<int>)'

// RWBuffer copy constructor
// CHECK-NEXT: CXXConstructExpr {{.*}} 'RWBuffer<int>':'hlsl::RWBuffer<int>' 'void (const hlsl::RWBuffer<int> &)'
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'const RWBuffer<int>':'const hlsl::RWBuffer<int>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const RWBuffer<int>':'const hlsl::RWBuffer<int>' <NoOp>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'RWBuffer<int>':'hlsl::RWBuffer<int>' <ConstructorConversion>

// RWBuffer heap info constructor
// CHECK-NEXT: CXXConstructExpr {{.*}} 'RWBuffer<int>':'hlsl::RWBuffer<int>' 'void (hlsl::__hlsl_heap_resource_info)'

// __hlsl_heap_resource_info copy constructor
// CHECK-NEXT: CXXConstructExpr {{.*}} 'hlsl::__hlsl_heap_resource_info' 'void (__hlsl_heap_resource_info &&) noexcept' elidable

// Indexing into ResourceDescriptorHeap
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} '__hlsl_heap_resource_info' xvalue
// CHECK-NEXT: CXXOperatorCallExpr {{.*}} '__hlsl_heap_resource_info' '[]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__hlsl_heap_resource_info (*)(uint32_t)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} '__hlsl_heap_resource_info (uint32_t)' lvalue CXXMethod {{.*}} 'operator[]' '__hlsl_heap_resource_info (uint32_t)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'hlsl::__hlsl_resource_descriptor_heap_struct' lvalue <AddressSpaceConversion>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl_private __hlsl_resource_descriptor_heap_struct' lvalue Var {{.*}} 'ResourceDescriptorHeap' 'hlsl_private __hlsl_resource_descriptor_heap_struct'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'

  useBuffer(ResourceDescriptorHeap[Index]);

// CHECK: VarDecl {{.*}} Sampler 'SamplerState' cinit
// CHECK-NEXT: ExprWithCleanups {{.*}} 'SamplerState'

// SamplerState copy constructor
// CHECK-NEXT: CXXConstructExpr {{.*}} <col:16, col:53> 'SamplerState' 'void (const hlsl::SamplerState &)' elidable
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} <col:26, col:53> 'const SamplerState' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} <col:26, col:53> 'const SamplerState' <NoOp>
// CHECK-NEXT: ImplicitCastExpr {{.*}} <col:26, col:53> 'SamplerState' <ConstructorConversion>

// SamplerState heap info constructor
// CHECK-NEXT: CXXConstructExpr {{.*}} <col:26, col:53> 'SamplerState' 'void (hlsl::__hlsl_heap_sampler_info)'

// __hlsl_heap_resource_info copy constructor
// CHECK-NEXT: CXXConstructExpr {{.*}} <col:26, col:53> 'hlsl::__hlsl_heap_sampler_info' 'void (__hlsl_heap_sampler_info &&) noexcept' elidable

// Indexing into SamplerDescriptorHeap
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} <col:26, col:53> '__hlsl_heap_sampler_info' xvalue
// CHECK-NEXT: CXXOperatorCallExpr {{.*}} <col:26, col:53> '__hlsl_heap_sampler_info' '[]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} <col:47, col:53> '__hlsl_heap_sampler_info (*)(uint32_t)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} <col:47, col:53> '__hlsl_heap_sampler_info (uint32_t)' lvalue CXXMethod {{.*}} 'operator[]' '__hlsl_heap_sampler_info (uint32_t)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} <col:26> 'hlsl::__hlsl_sampler_descriptor_heap_struct' lvalue <AddressSpaceConversion>
// CHECK-NEXT: DeclRefExpr {{.*}} <col:26> 'hlsl_private __hlsl_sampler_descriptor_heap_struct' lvalue Var {{.*}} 'SamplerDescriptorHeap' 'hlsl_private __hlsl_sampler_descriptor_heap_struct'
// CHECK-NEXT: ImplicitCastExpr {{.*}} <col:48> 'unsigned int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} <col:48> 'unsigned int' lvalue ParmVar {{.*}} 'Index' 'unsigned int'

  SamplerState Sampler = SamplerDescriptorHeap[Index];
}
