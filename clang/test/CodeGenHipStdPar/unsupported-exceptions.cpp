// REQUIRES: amdgpu-registered-target
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu --hipstdpar -x hip \
// RUN:   -fcuda-is-device -fcxx-exceptions -fexceptions \
// RUN:   -mllvm -amdgpu-enable-hipstdpar -emit-llvm -o - -DREACHABLE %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REACHABLE
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu --hipstdpar -x hip \
// RUN:   -fcuda-is-device -fcxx-exceptions -fexceptions \
// RUN:   -emit-llvm -o - -DTRY_CATCH %s \
// RUN:   | FileCheck %s --check-prefix=TRY-IR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -aux-triple amdgcn-amd-amdhsa --hipstdpar -x hip \
// RUN:   -fcxx-exceptions -fexceptions -emit-llvm -o - -DTRY_CATCH %s \
// RUN:   | FileCheck %s --check-prefix=HOST-IR
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu --hipstdpar -x hip \
// RUN:   -fcuda-is-device -fcxx-exceptions -fexceptions \
// RUN:   -mllvm -amdgpu-enable-hipstdpar -emit-llvm -o - -DTRY_CATCH %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TRY-CATCH
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu --hipstdpar -x hip \
// RUN:   -fcuda-is-device -fcxx-exceptions -fexceptions \
// RUN:   -mllvm -amdgpu-enable-hipstdpar -emit-llvm -o - -DFUNCTION_TRY %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FUNCTION-TRY
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu --hipstdpar -x hip \
// RUN:   -fcuda-is-device -fcxx-exceptions -fexceptions \
// RUN:   -mllvm -amdgpu-enable-hipstdpar -emit-llvm -o - \
// RUN:   -DCONSTRUCTOR_FUNCTION_TRY %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CONSTRUCTOR-FUNCTION-TRY
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu --hipstdpar -x hip \
// RUN:   -fcuda-is-device -fcxx-exceptions -fexceptions \
// RUN:   -mllvm -amdgpu-enable-hipstdpar -emit-llvm -o - \
// RUN:   -DDESTRUCTOR_FUNCTION_TRY %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DESTRUCTOR-FUNCTION-TRY
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu --hipstdpar -x hip \
// RUN:   -fcuda-is-device -fcxx-exceptions -fexceptions \
// RUN:   -mllvm -amdgpu-enable-hipstdpar -emit-llvm -o - -DBAD_CAST %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-CAST
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu --hipstdpar -x hip \
// RUN:   -fcuda-is-device -fcxx-exceptions -fexceptions \
// RUN:   -mllvm -amdgpu-enable-hipstdpar -emit-llvm -o - -DBAD_TYPEID %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-TYPEID
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu --hipstdpar -x hip \
// RUN:   -fcuda-is-device -fcxx-exceptions -fexceptions \
// RUN:   -mllvm -amdgpu-enable-hipstdpar -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=UNREACHABLE

#define __global__ __attribute__((global))

#if defined(BAD_CAST) || defined(BAD_TYPEID)
namespace std {
class type_info;
}

struct Base {
  virtual ~Base();
};
struct Derived : Base {};
#endif

#if defined(BAD_CAST)
Derived &badCast(Base &B) { return dynamic_cast<Derived &>(B); }

__global__ void kernel(Base *B) { (void)badCast(*B); }

// BAD-CAST: error: Accelerator does not support C++ exception handling.
#elif defined(BAD_TYPEID)
const std::type_info &badTypeid(Base *B) { return typeid(*B); }

__global__ void kernel(Base *B) { (void)badTypeid(B); }

// BAD-TYPEID: error: Accelerator does not support C++ exception handling.
#else
void may_throw();

void throwing_helper() { throw 1; }

void try_helper() {
  try {
    may_throw();
  } catch (...) {
  }
}

void function_try_helper() try {
  may_throw();
} catch (...) {
}

struct ConstructorFunctionTry {
  ConstructorFunctionTry();
};

ConstructorFunctionTry::ConstructorFunctionTry() try {
  may_throw();
} catch (...) {
}

struct DestructorFunctionTry {
  ~DestructorFunctionTry();
};

DestructorFunctionTry::~DestructorFunctionTry() try {
  may_throw();
} catch (...) {
}

__global__ void kernel() {
#ifdef REACHABLE
  throwing_helper();
#elif defined(TRY_CATCH)
  try_helper();
#elif defined(FUNCTION_TRY)
  function_try_helper();
#elif defined(CONSTRUCTOR_FUNCTION_TRY)
  ConstructorFunctionTry value;
#elif defined(DESTRUCTOR_FUNCTION_TRY)
  DestructorFunctionTry value;
#endif
}

// REACHABLE: error: Accelerator does not support C++ exception handling.
// TRY-CATCH: error: Accelerator does not support C++ exception handling.
// FUNCTION-TRY: error: Accelerator does not support C++ exception handling.
// CONSTRUCTOR-FUNCTION-TRY: error: Accelerator does not support C++ exception handling.
// DESTRUCTOR-FUNCTION-TRY: error: Accelerator does not support C++ exception handling.

// TRY-IR-LABEL: define{{.*}} void @_Z10try_helperv()
// TRY-IR: call void @__CXX_EXCEPTION__hipstdpar_unsupported()

// HOST-IR-NOT: @__CXX_EXCEPTION__hipstdpar_unsupported
// HOST-IR-LABEL: define{{.*}} void @_Z10try_helperv()
// HOST-IR-NOT: @__CXX_EXCEPTION__hipstdpar_unsupported
// HOST-IR: invoke void @_Z9may_throwv()
// HOST-IR-NOT: @__CXX_EXCEPTION__hipstdpar_unsupported

// UNREACHABLE-NOT: @__cxa_throw
// UNREACHABLE-NOT: @__CXX_EXCEPTION__hipstdpar_unsupported
// UNREACHABLE-NOT: @_Z15throwing_helperv
// UNREACHABLE-NOT: @_Z10try_helperv
// UNREACHABLE-NOT: @_Z19function_try_helperv
// UNREACHABLE: define{{.*}} amdgpu_kernel void @_Z6kernelv()
#endif
