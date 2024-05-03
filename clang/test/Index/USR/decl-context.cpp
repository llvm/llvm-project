// RUN: c-index-test core -print-source-symbols -- -std=c++20 %s | FileCheck %s

namespace ns {
namespace {
struct Foo {};
// CHECK: [[@LINE-1]]:8 | struct/C | Foo | c:decl-context.cpp@N@ns@aN@S@Foo
}
}
namespace ns2 {
namespace {
struct Foo {};
// CHECK: [[@LINE-1]]:8 | struct/C | Foo | c:decl-context.cpp@N@ns2@aN@S@Foo
}
}
