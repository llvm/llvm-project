// RUN: %check_clang_tidy %s llvmlibc-implementation-in-namespace %t

#define MACRO_A "defining macros outside namespace is valid"

class ClassB;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration must be enclosed within the 'LIBC_NAMESPACE' namespace
struct StructC {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: declaration must be enclosed within the 'LIBC_NAMESPACE' namespace
char *VarD = MACRO_A;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration must be enclosed within the 'LIBC_NAMESPACE' namespace
typedef int typeE;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: declaration must be enclosed within the 'LIBC_NAMESPACE' namespace
void funcF() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: declaration must be enclosed within the 'LIBC_NAMESPACE' namespace

namespace outer_most {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: the outermost namespace should be the 'LIBC_NAMESPACE' macro
 class A {};
}

// Wrapped in anonymous namespace.
namespace {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: declaration must be enclosed within the 'LIBC_NAMESPACE' namespace
 class A {};
}

namespace namespaceG {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: the outermost namespace should be the 'LIBC_NAMESPACE' macro
namespace __llvm_libc {
namespace namespaceH {
class ClassB;
} // namespace namespaceH
struct StructC {};
} // namespace __llvm_libc
char *VarD = MACRO_A;
typedef int typeE;
void funcF() {}
} // namespace namespaceG

// Wrapped in macro namespace but with an incorrect name
#define LIBC_NAMESPACE custom_namespace
namespace LIBC_NAMESPACE {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: the 'LIBC_NAMESPACE' macro should start with '__llvm_libc'
namespace namespaceH {
class ClassB;
} // namespace namespaceH
} // namespace LIBC_NAMESPACE


// Wrapped in macro namespace with a valid name, LIBC_NAMESPACE starts with '__llvm_libc'
#undef LIBC_NAMESPACE
#define LIBC_NAMESPACE __llvm_libc_xyz
namespace LIBC_NAMESPACE {
namespace namespaceI {
class ClassB;
} // namespace namespaceI
struct StructC {};
char *VarD = MACRO_A;
typedef int typeE;
void funcF() {}
extern "C" void extern_funcJ() {}
} // namespace LIBC_NAMESPACE
