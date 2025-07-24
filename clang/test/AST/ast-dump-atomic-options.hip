// RUN: %clang_cc1 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fcuda-is-device %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fcuda-is-device %s \
// RUN:   -fatomic-fine-grained-memory -fatomic-ignore-denormal-mode \
// RUN:   | FileCheck %s

#include "Inputs/cuda.h"

__device__ __host__ void test_default(float *a) {
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// CHECK: |-FunctionDecl {{.*}} test_default 'void (float *)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used a 'float *'
// CHECK-NEXT: | | `-typeDetails: PointerType {{.*}} 'float *'
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | |-CompoundStmt {{.*}} 
// CHECK-NEXT: | | `-AtomicExpr {{.*}} 'float' __scoped_atomic_fetch_add
// CHECK-NEXT: | |   |-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK-NEXT: | |   | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'a' 'float *'
// CHECK-NEXT: | |   |-IntegerLiteral {{.*}} <<built-in>:7:26> 'int' 0
// CHECK-NEXT: | |   |-ImplicitCastExpr {{.*}} 
// CHECK-NEXT: | |   | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: | |   `-IntegerLiteral {{.*}} <<built-in>:13:31> 'int' 0
// CHECK-NEXT: | |-attrDetails: CUDADeviceAttr {{.*}} 
// CHECK-NEXT: | `-attrDetails: CUDAHostAttr {{.*}} 
// CHECK-NEXT: |-LinkageSpecDecl {{.*}} implicit C
// CHECK-NEXT: | `-FunctionDecl {{.*}} implicit used __scoped_atomic_fetch_add 'void (...)' extern
// CHECK-NEXT: |   `-attrDetails: BuiltinAttr {{.*}} <<invalid sloc>> Implicit 72

__device__ __host__ void test_one(float *a) {
  [[clang::atomic(no_remote_memory)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

// CHECK: |-FunctionDecl {{.*}} test_one 'void (float *)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used a 'float *'
// CHECK-NEXT: | | `-typeDetails: PointerType {{.*}} 'float *'
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | |-CompoundStmt {{.*}} 
// CHECK-NEXT: | | `-AttributedStmt {{.*}} 
// CHECK-NEXT: | |   |-attrDetails: AtomicAttr {{.*}} no_remote_memory
// CHECK-NEXT: | |   `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |     `-AtomicExpr {{.*}} 'float' __scoped_atomic_fetch_add
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'a' 'float *'
// CHECK-NEXT: | |       |-IntegerLiteral {{.*}} <<built-in>:7:26> 'int' 0
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 
// CHECK-NEXT: | |       | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: | |       `-IntegerLiteral {{.*}} <<built-in>:13:31> 'int' 0
// CHECK-NEXT: | |-attrDetails: CUDADeviceAttr {{.*}} 
// CHECK-NEXT: | `-attrDetails: CUDAHostAttr {{.*}} 

__device__ __host__ void test_two(float *a) {
  [[clang::atomic(remote_memory, ignore_denormal_mode)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

// CHECK: |-FunctionDecl {{.*}} test_two 'void (float *)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used a 'float *'
// CHECK-NEXT: | | `-typeDetails: PointerType {{.*}} 'float *'
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | |-CompoundStmt {{.*}} 
// CHECK-NEXT: | | `-AttributedStmt {{.*}} 
// CHECK-NEXT: | |   |-attrDetails: AtomicAttr {{.*}} remote_memory ignore_denormal_mode
// CHECK-NEXT: | |   `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |     `-AtomicExpr {{.*}} 'float' __scoped_atomic_fetch_add
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'a' 'float *'
// CHECK-NEXT: | |       |-IntegerLiteral {{.*}} <<built-in>:7:26> 'int' 0
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 
// CHECK-NEXT: | |       | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: | |       `-IntegerLiteral {{.*}} <<built-in>:13:31> 'int' 0
// CHECK-NEXT: | |-attrDetails: CUDADeviceAttr {{.*}} 
// CHECK-NEXT: | `-attrDetails: CUDAHostAttr {{.*}} 


__device__ __host__ void test_three(float *a) {
  [[clang::atomic(no_remote_memory, fine_grained_memory, no_ignore_denormal_mode)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

// CHECK: |-FunctionDecl {{.*}} test_three 'void (float *)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used a 'float *'
// CHECK-NEXT: | | `-typeDetails: PointerType {{.*}} 'float *'
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | |-CompoundStmt {{.*}} 
// CHECK-NEXT: | | `-AttributedStmt {{.*}} 
// CHECK-NEXT: | |   |-attrDetails: AtomicAttr {{.*}} no_remote_memory fine_grained_memory no_ignore_denormal_mode
// CHECK-NEXT: | |   `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |     `-AtomicExpr {{.*}} 'float' __scoped_atomic_fetch_add
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'a' 'float *'
// CHECK-NEXT: | |       |-IntegerLiteral {{.*}} <<built-in>:7:26> 'int' 0
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 
// CHECK-NEXT: | |       | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: | |       `-IntegerLiteral {{.*}} <<built-in>:13:31> 'int' 0
// CHECK-NEXT: | |-attrDetails: CUDADeviceAttr {{.*}} 
// CHECK-NEXT: | `-attrDetails: CUDAHostAttr {{.*}} 

__device__ __host__ void test_duplicate(float *a) {
  [[clang::atomic(no_remote_memory, no_remote_memory)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

// CHECK: |-FunctionDecl {{.*}} test_duplicate 'void (float *)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used a 'float *'
// CHECK-NEXT: | | `-typeDetails: PointerType {{.*}} 'float *'
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | |-CompoundStmt {{.*}} 
// CHECK-NEXT: | | `-AttributedStmt {{.*}} 
// CHECK-NEXT: | |   |-attrDetails: AtomicAttr {{.*}} no_remote_memory no_remote_memory
// CHECK-NEXT: | |   `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |     `-AtomicExpr {{.*}} 'float' __scoped_atomic_fetch_add
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'a' 'float *'
// CHECK-NEXT: | |       |-IntegerLiteral {{.*}} <<built-in>:7:26> 'int' 0
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 
// CHECK-NEXT: | |       | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: | |       `-IntegerLiteral {{.*}} <<built-in>:13:31> 'int' 0
// CHECK-NEXT: | |-attrDetails: CUDADeviceAttr {{.*}} 
// CHECK-NEXT: | `-attrDetails: CUDAHostAttr {{.*}} 

__device__ __host__ void test_conflict(float *a) {
  [[clang::atomic(no_remote_memory, remote_memory)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

// CHECK: |-FunctionDecl {{.*}} test_conflict 'void (float *)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used a 'float *'
// CHECK-NEXT: | | `-typeDetails: PointerType {{.*}} 'float *'
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | |-CompoundStmt {{.*}} 
// CHECK-NEXT: | | `-AttributedStmt {{.*}} 
// CHECK-NEXT: | |   |-attrDetails: AtomicAttr {{.*}} no_remote_memory remote_memory
// CHECK-NEXT: | |   `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |     `-AtomicExpr {{.*}} 'float' __scoped_atomic_fetch_add
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'a' 'float *'
// CHECK-NEXT: | |       |-IntegerLiteral {{.*}} <<built-in>:7:26> 'int' 0
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 
// CHECK-NEXT: | |       | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: | |       `-IntegerLiteral {{.*}} <<built-in>:13:31> 'int' 0
// CHECK-NEXT: | |-attrDetails: CUDADeviceAttr {{.*}} 
// CHECK-NEXT: | `-attrDetails: CUDAHostAttr {{.*}} 

__device__ __host__ void test_multiple_attrs(float *a) {
  [[clang::atomic(no_remote_memory)]] [[clang::atomic(remote_memory)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}


// CHECK: |-FunctionDecl {{.*}} test_multiple_attrs 'void (float *)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used a 'float *'
// CHECK-NEXT: | | `-typeDetails: PointerType {{.*}} 'float *'
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | |-CompoundStmt {{.*}} 
// CHECK-NEXT: | | `-AttributedStmt {{.*}} 
// CHECK-NEXT: | |   |-attrDetails: AtomicAttr {{.*}} no_remote_memory
// CHECK-NEXT: | |   |-attrDetails: AtomicAttr {{.*}} remote_memory
// CHECK-NEXT: | |   `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |     `-AtomicExpr {{.*}} 'float' __scoped_atomic_fetch_add
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'a' 'float *'
// CHECK-NEXT: | |       |-IntegerLiteral {{.*}} <<built-in>:7:26> 'int' 0
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 
// CHECK-NEXT: | |       | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: | |       `-IntegerLiteral {{.*}} <<built-in>:13:31> 'int' 0
// CHECK-NEXT: | |-attrDetails: CUDADeviceAttr {{.*}} 
// CHECK-NEXT: | `-attrDetails: CUDAHostAttr {{.*}} 

__device__ __host__ void test_nested(float *a) {
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  [[clang::atomic(remote_memory, fine_grained_memory, no_ignore_denormal_mode)]] {
    __scoped_atomic_fetch_max(a, 2, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
    [[clang::atomic(no_remote_memory)]] {
      __scoped_atomic_fetch_min(a, 3, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_WRKGRP);
    }
    [[clang::atomic(no_fine_grained_memory)]] {
      __scoped_atomic_fetch_sub(a, 4, __ATOMIC_RELEASE, __MEMORY_SCOPE_WVFRNT);
    }
  }
}

// CHECK: |-FunctionDecl {{.*}} test_nested 'void (float *)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used a 'float *'
// CHECK-NEXT: | | `-typeDetails: PointerType {{.*}} 'float *'
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: | |-CompoundStmt {{.*}} 
// CHECK-NEXT: | | |-AtomicExpr {{.*}} 'float' __scoped_atomic_fetch_add
// CHECK-NEXT: | | | |-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK-NEXT: | | | | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'a' 'float *'
// CHECK-NEXT: | | | |-IntegerLiteral {{.*}} <<built-in>:7:26> 'int' 0
// CHECK-NEXT: | | | |-ImplicitCastExpr {{.*}} 
// CHECK-NEXT: | | | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: | | | `-IntegerLiteral {{.*}} <<built-in>:13:31> 'int' 0
// CHECK-NEXT: | | `-AttributedStmt {{.*}} 
// CHECK-NEXT: | |   |-attrDetails: AtomicAttr {{.*}} remote_memory fine_grained_memory no_ignore_denormal_mode
// CHECK-NEXT: | |   `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |     |-AtomicExpr {{.*}} 'float' __scoped_atomic_fetch_max
// CHECK-NEXT: | |     | |-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK-NEXT: | |     | | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'a' 'float *'
// CHECK-NEXT: | |     | |-IntegerLiteral {{.*}} <<built-in>:12:26> 'int' 5
// CHECK-NEXT: | |     | |-ImplicitCastExpr {{.*}} 
// CHECK-NEXT: | |     | | `-IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: | |     | `-IntegerLiteral {{.*}} <<built-in>:14:31> 'int' 1
// CHECK-NEXT: | |     |-AttributedStmt {{.*}} 
// CHECK-NEXT: | |     | |-attrDetails: AtomicAttr {{.*}} no_remote_memory
// CHECK-NEXT: | |     | `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |     |   `-AtomicExpr {{.*}} 'float' __scoped_atomic_fetch_min
// CHECK-NEXT: | |     |     |-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK-NEXT: | |     |     | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'a' 'float *'
// CHECK-NEXT: | |     |     |-IntegerLiteral {{.*}} <<built-in>:9:26> 'int' 2
// CHECK-NEXT: | |     |     |-ImplicitCastExpr {{.*}} 
// CHECK-NEXT: | |     |     | `-IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: | |     |     `-IntegerLiteral {{.*}} <<built-in>:15:31> 'int' 2
// CHECK-NEXT: | |     `-AttributedStmt {{.*}} 
// CHECK-NEXT: | |       |-attrDetails: AtomicAttr {{.*}} no_fine_grained_memory
// CHECK-NEXT: | |       `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |         `-AtomicExpr {{.*}} 'float' __scoped_atomic_fetch_sub
// CHECK-NEXT: | |           |-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK-NEXT: | |           | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'a' 'float *'
// CHECK-NEXT: | |           |-IntegerLiteral {{.*}} <<built-in>:10:26> 'int' 3
// CHECK-NEXT: | |           |-ImplicitCastExpr {{.*}} 
// CHECK-NEXT: | |           | `-IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT: | |           `-IntegerLiteral {{.*}} <<built-in>:16:31> 'int' 3
// CHECK-NEXT: | |-attrDetails: CUDADeviceAttr {{.*}} 
// CHECK-NEXT: | `-attrDetails: CUDAHostAttr {{.*}} 
// CHECK-NEXT: |-LinkageSpecDecl {{.*}} implicit C
// CHECK-NEXT: | `-FunctionDecl {{.*}} implicit used __scoped_atomic_fetch_max 'void (...)' extern
// CHECK-NEXT: |   `-attrDetails: BuiltinAttr {{.*}} <<invalid sloc>> Implicit 74
// CHECK-NEXT: |-LinkageSpecDecl {{.*}} implicit C
// CHECK-NEXT: | `-FunctionDecl {{.*}} implicit used __scoped_atomic_fetch_min 'void (...)' extern
// CHECK-NEXT: |   `-attrDetails: BuiltinAttr {{.*}} <<invalid sloc>> Implicit 75
// CHECK-NEXT: |-LinkageSpecDecl {{.*}} implicit C
// CHECK-NEXT: | `-FunctionDecl {{.*}} implicit used __scoped_atomic_fetch_sub 'void (...)' extern
// CHECK-NEXT: |   `-attrDetails: BuiltinAttr {{.*}} <<invalid sloc>> Implicit 78


template<typename T>
__device__ __host__ void test_template(T *a) {
  [[clang::atomic(no_remote_memory, fine_grained_memory, no_ignore_denormal_mode)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __host__ void test_template_caller() {
  float *p;
  test_template(p);
}

// CHECK: |-FunctionTemplateDecl {{.*}} test_template
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl {{.*}} test_template 'void (T *)'
// CHECK-NEXT: |   |-ParmVarDecl {{.*}} referenced a 'T *'
// CHECK-NEXT: |   | `-typeDetails: PointerType {{.*}} 'T *' dependent
// CHECK-NEXT: |   |   `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
// CHECK-NEXT: |   |     `-TemplateTypeParm {{.*}} 'T'
// CHECK-NEXT: |   |-CompoundStmt {{.*}} 
// CHECK-NEXT: |   | `-AttributedStmt {{.*}} 
// CHECK-NEXT: |   |   |-attrDetails: AtomicAttr {{.*}} no_remote_memory fine_grained_memory no_ignore_denormal_mode
// CHECK-NEXT: |   |   `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   |     `-CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: |   |       |-DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__scoped_atomic_fetch_add' 'void (...)'
// CHECK-NEXT: |   |       |-DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'a' 'T *'
// CHECK-NEXT: |   |       |-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: |   |       |-IntegerLiteral {{.*}} <<built-in>:7:26> 'int' 0
// CHECK-NEXT: |   |       `-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |   |-attrDetails: CUDADeviceAttr {{.*}} 
// CHECK-NEXT: |   `-attrDetails: CUDAHostAttr {{.*}} 
// CHECK: `-FunctionDecl {{.*}} test_template_caller 'void ()'
// CHECK-NEXT:   |-CompoundStmt {{.*}} 
// CHECK-NEXT:   | |-DeclStmt {{.*}} 
// CHECK-NEXT:   | | `-VarDecl {{.*}} used p 'float *'
// CHECK-NEXT:   | |   `-typeDetails: PointerType {{.*}} 'float *'
// CHECK-NEXT:   | |     `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT:   | `-CallExpr {{.*}} 'void'
// CHECK-NEXT:   |   |-ImplicitCastExpr {{.*}} 'void (*)(float *)' <FunctionToPointerDecay>
// CHECK-NEXT:   |   | `-DeclRefExpr {{.*}} 'void (float *)' lvalue Function {{.*}} 'test_template' 'void (float *)' (FunctionTemplate {{.*}} 'test_template')
// CHECK-NEXT:   |   `-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK-NEXT:   |     `-DeclRefExpr {{.*}} 'float *' lvalue Var {{.*}} 'p' 'float *'
// CHECK-NEXT:   |-attrDetails: CUDADeviceAttr {{.*}}
// CHECK-NEXT:   `-attrDetails: CUDAHostAttr {{.*}}
