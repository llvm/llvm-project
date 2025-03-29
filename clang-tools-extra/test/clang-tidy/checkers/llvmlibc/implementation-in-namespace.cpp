// RUN: %check_clang_tidy %s llvmlibc-implementation-in-namespace %t

#define MACRO_A "defining macros outside namespace is valid"

class ClassB;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration must be enclosed within the 'LIBC_NAMESPACE_DECL' namespace
struct StructC {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: declaration must be enclosed within the 'LIBC_NAMESPACE_DECL' namespace
const char *VarD = MACRO_A;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: declaration must be enclosed within the 'LIBC_NAMESPACE_DECL' namespace
typedef int typeE;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: declaration must be enclosed within the 'LIBC_NAMESPACE_DECL' namespace
void funcF() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: declaration must be enclosed within the 'LIBC_NAMESPACE_DECL' namespace

namespace outer_most {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: the outermost namespace should be the 'LIBC_NAMESPACE_DECL' macro
 class A {};
}

// Wrapped in anonymous namespace.
namespace {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: declaration must be enclosed within the 'LIBC_NAMESPACE_DECL' namespace
 class A {};
}

namespace namespaceG {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: the outermost namespace should be the 'LIBC_NAMESPACE_DECL' macro
namespace __llvm_libc {
namespace namespaceH {
class ClassB;
} // namespace namespaceH
struct StructC {};
} // namespace __llvm_libc
const char *VarD = MACRO_A;
typedef int typeE;
void funcF() {}
} // namespace namespaceG

// Wrapped in macro namespace but with an incorrect name
#define LIBC_NAMESPACE_DECL [[gnu::visibility("hidden")]] custom_namespace
namespace LIBC_NAMESPACE_DECL {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: the 'LIBC_NAMESPACE_DECL' macro expansion should start with '__llvm_libc'

namespace namespaceH {
class ClassB;
} // namespace namespaceH
} // namespace LIBC_NAMESPACE_DECL


// Wrapped in macro namespace with a valid name, LIBC_NAMESPACE_DECL starts with '__llvm_libc'
#undef LIBC_NAMESPACE_DECL
#define LIBC_NAMESPACE_DECL [[gnu::visibility("hidden")]] __llvm_libc_xyz
namespace LIBC_NAMESPACE_DECL {
namespace namespaceI {
class ClassB;
} // namespace namespaceI
struct StructC {};
const char *VarD = MACRO_A;
typedef int typeE;
void funcF() {}
extern "C" void extern_funcJ() {}
} // namespace LIBC_NAMESPACE_DECL
