// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -gno-column-info -triple x86_64-apple-darwin10 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -gno-column-info -triple x86_64-apple-darwin10 -std=c++98 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -gno-column-info -triple x86_64-apple-darwin10 -std=c++11 %s -o - | FileCheck %s

// Check the line numbers for cleanup code with EH in combination with
// simple return expressions.

// CHECK: define {{.*}}foo
// CHECK: call void @_ZN1CD1Ev(%class.C* {{.*}}){{( #[0-9])?}}, !dbg ![[RET:[0-9]+]]
// CHECK: ret i32 0, !dbg ![[RET]]

// CHECK: define {{.*}}bar
// CHECK: ret void, !dbg ![[RETBAR:[0-9]+]]

// CHECK: define {{.*}}baz
// CHECK: ret void, !dbg ![[RETBAZ:[0-9]+]]

class C {
public:
  ~C() {}
  int i;
};

int foo()
{
  C c;
  c.i = 42;
  return 0;
  // This breakpoint should be at/before the cleanup code.
  // CHECK: ![[RET]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
}

void bar()
{
  if (!foo())
    // CHECK: {{.*}} = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
    return;

  if (foo()) {
    C c;
    c.i = foo();
  }
  // Clang creates only a single ret instruction. Make sure it is at a useful line.
  // CHECK: ![[RETBAR]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
}

void baz()
{
  if (!foo())
    // CHECK: ![[SCOPE1:.*]] = distinct !DILexicalBlock({{.*}}, line: [[@LINE-1]])
    // CHECK: {{.*}} = !DILocation(line: [[@LINE+1]], scope: ![[SCOPE1]])
    return;

  if (foo()) {
    // no cleanup
    // CHECK: {{.*}} = !DILocation(line: [[@LINE+2]], scope: ![[SCOPE2:.*]])
    // CHECK: ![[SCOPE2]] = distinct !DILexicalBlock({{.*}}, line: [[@LINE-3]])
    return;
  }
  // CHECK: ![[RETBAZ]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
}
