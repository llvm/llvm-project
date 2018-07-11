// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-macosx10.9.0 -munwind-tables -std=c++11 -fcxx-exceptions -fexceptions %s -o - | FileCheck -allow-deprecated-dag-overlap %s

// Test that emitting a landing pad does not affect the line table
// entries for the code that triggered it.

// CHECK: call void @llvm.dbg.declare
// CHECK: call void @llvm.dbg.declare(metadata {{.*}}, metadata ![[CURRENT_ADDR:.*]], metadata !{{.*}}), !dbg ![[DBG1:.*]]
// CHECK: unwind label %{{.*}}, !dbg ![[DBG1]]
// CHECK: store i64 %{{.*}}, i64* %current_address, align 8, !dbg ![[DBG4:.*]]
// CHECK-NEXT: call void @llvm.dbg.declare(metadata {{.*}}, metadata ![[FOUND_IT:.*]], metadata !{{.*}}), !dbg ![[DBG2:.*]]
// CHECK: = landingpad
// CHECK-NEXT: cleanup, !dbg ![[DBG3:.*]]
// CHECK-DAG: ![[CURRENT_ADDR]] = {{.*}}name: "current_address"
// CHECK-DAG: ![[FOUND_IT]] = {{.*}}name: "found_it"
// CHECK-DAG: ![[DBG1]] = !DILocation(line: 256,
// CHECK-DAG: ![[DBG2]] = !DILocation(line: 257,
// CHECK-DAG: ![[DBG3]] = !DILocation(line: 268,
// CHECK-DAG: ![[DBG4]] = !DILocation(line: 256,
typedef unsigned long long uint64_t;
template<class _Tp> class shared_ptr {
public:
  typedef _Tp element_type;
  element_type* __ptr_;
  ~shared_ptr();
  element_type* operator->() const noexcept {return __ptr_;}
};
class Context {
public:
    uint64_t GetIt();
};
class Foo
{
    bool bar();
    virtual shared_ptr<Context> GetContext () = 0;
};
# 253 "Foo.cpp" 3
bool
Foo::bar ()
{
  uint64_t current_address = GetContext()->GetIt();
  bool found_it = false;
# 267 "Foo.cpp" 3
  return found_it;
}
