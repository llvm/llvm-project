// RUN: %check_clang_tidy %s llvmlibc-implementation-in-namespace %t

#define MACRO_A "defining macros outside namespace is valid"

class ClassB;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration must be declared within the '__llvm_libc' namespace
struct StructC {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: declaration must be declared within the '__llvm_libc' namespace
char *VarD = MACRO_A;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration must be declared within the '__llvm_libc' namespace
typedef int typeE;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: declaration must be declared within the '__llvm_libc' namespace
void funcF() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: declaration must be declared within the '__llvm_libc' namespace

namespace namespaceG {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: '__llvm_libc' needs to be the outermost namespace
namespace __llvm_libc{
namespace namespaceH {
class ClassB;
} // namespace namespaceH
struct StructC {};
} // namespace __llvm_libc
char *VarD = MACRO_A;
typedef int typeE;
void funcF() {}
} // namespace namespaceG

// Wrapped in correct namespace.
namespace __llvm_libc {
// Namespaces within __llvim_libc namespace allowed.
namespace namespaceI {
class ClassB;
} // namespace namespaceI
struct StructC {};
char *VarD = MACRO_A;
typedef int typeE;
void funcF() {}
extern "C" void extern_funcJ() {}
} // namespace __llvm_libc
