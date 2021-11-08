// RUN: %clang_cc1 -triple x86_64-unknown-windows -emit-llvm -target-cpu core2 -o - %s | FileCheck %s

#define SWIFTCALL __attribute__((swiftcall))
#define OUT __attribute__((swift_indirect_result))
#define ERROR __attribute__((swift_error_result))
#define CONTEXT __attribute__((swift_context))

/*****************************************************************************/
/****************************** PARAMETER ABIS *******************************/
/*****************************************************************************/

// Swift doesn't use inalloca like windows x86 normally does.
struct NonTrivial {
  NonTrivial();
  NonTrivial(const NonTrivial &);
  int o;
};

SWIFTCALL int receiveNonTrivial(NonTrivial o) { return o.o; }

// CHECK-LABEL: define dso_local swiftcc i32 @"?receiveNonTrivial@@YSHUNonTrivial@@@Z"(%struct.NonTrivial* %o)

int passNonTrivial() {
  return receiveNonTrivial({});
}

// CHECK-LABEL: define dso_local i32 @"?passNonTrivial@@YAHXZ"()
// CHECK-NOT: stacksave
// CHECK: call swiftcc i32 @"?receiveNonTrivial@@YSHUNonTrivial@@@Z"(%struct.NonTrivial* %{{.*}})
