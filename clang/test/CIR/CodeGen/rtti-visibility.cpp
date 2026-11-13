// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

namespace std {
class type_info {
public:
  virtual ~type_info();
};
} // namespace std

struct __attribute__((visibility("hidden"))) Hidden {
  virtual void f();
};
struct __attribute__((visibility("protected"))) Protected {
  virtual void f();
};
struct Default {
  virtual void f();
};

const void *hidden = &typeid(Hidden);
const void *prot = &typeid(Protected);
const void *def = &typeid(Default);

void Hidden::f() {}
// CIR-DAG: cir.global hidden constant external dso_local @_ZTV6Hidden =
// CIR-DAG: cir.global hidden constant external dso_local @_ZTS6Hidden =
// CIR-DAG: cir.global hidden constant external dso_local @_ZTI6Hidden =

// LLVM-DAG: @_ZTV6Hidden = hidden constant { [3 x ptr] }
// LLVM-DAG: @_ZTS6Hidden = hidden constant [8 x i8]
// LLVM-DAG: @_ZTI6Hidden = hidden constant { ptr, ptr }
void Protected::f() {}
// CIR-DAG: cir.global protected constant external dso_local @_ZTV9Protected =
// CIR-DAG: cir.global protected constant external dso_local @_ZTS9Protected =
// CIR-DAG: cir.global protected constant external dso_local @_ZTI9Protected =

// LLVM-DAG: @_ZTV9Protected = protected constant { [3 x ptr] }
// LLVM-DAG: @_ZTS9Protected = protected constant [11 x i8]
// LLVM-DAG: @_ZTI9Protected = protected constant { ptr, ptr }
void Default::f() {}
// CIR-DAG: cir.global constant external @_ZTV7Default =
// CIR-DAG: cir.global constant external @_ZTS7Default =
// CIR-DAG: cir.global constant external @_ZTI7Default =

// LLVM-DAG: @_ZTV7Default = constant { [3 x ptr] }
// LLVM-DAG: @_ZTS7Default = constant [9 x i8]
// LLVM-DAG: @_ZTI7Default = constant { ptr, ptr }

