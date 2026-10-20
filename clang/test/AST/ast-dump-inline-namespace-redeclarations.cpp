// RUN: %clang_cc1 -std=c++11 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -include-pch %t -ast-dump-all -x c++ /dev/null | FileCheck %s

namespace reopen {
// CHECK: NamespaceDecl [[V1:0x[^ ]+]] {{.*}} v1 inline
inline namespace v1 {
// CHECK: NamespaceDecl [[NS:0x[^ ]+]] {{.*}} ns
namespace ns {
struct Y;
}
}
// CHECK: NamespaceDecl {{.*}} parent [[V1]] prev [[NS]] {{.*}} ns
namespace ns {
struct Y {};
}
}

namespace qualified {
// CHECK: NamespaceDecl [[QV1:0x[^ ]+]] {{.*}} v1 inline
inline namespace v1 {
// CHECK: CXXRecordDecl [[X:0x[^ ]+]] {{.*}} struct X
struct X;
}
}
namespace alias = qualified;
// CHECK: CXXRecordDecl {{.*}} parent [[QV1]] prev [[X]] {{.*}} struct X definition
struct alias::X {};

// Ordinary namespace redeclarations keep the current enclosing declaration
// as both their lexical and semantic context.
namespace ordinary {
// CHECK: NamespaceDecl [[NESTED:0x[^ ]+]] {{.*}} nested
namespace nested {}
}
namespace ordinary {
// CHECK: NamespaceDecl {{0x[^ ]+}} prev [[NESTED]] {{.*}} nested
namespace nested {}
}
