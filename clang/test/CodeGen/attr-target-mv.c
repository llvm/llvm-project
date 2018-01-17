// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
int __attribute__((target("sse4.2"))) foo(void) { return 0; }
int __attribute__((target("arch=sandybridge"))) foo(void);
int __attribute__((target("arch=ivybridge"))) foo(void) {return 1;}
int __attribute__((target("default"))) foo(void) { return 2; }

int bar() {
  return foo();
}

inline int __attribute__((target("sse4.2"))) foo_inline(void) { return 0; }
inline int __attribute__((target("arch=sandybridge"))) foo_inline(void);
inline int __attribute__((target("arch=ivybridge"))) foo_inline(void) {return 1;}
inline int __attribute__((target("default"))) foo_inline(void) { return 2; }

int bar2() {
  return foo_inline();
}

inline __attribute__((target("default"))) void foo_decls(void);
inline __attribute__((target("sse4.2"))) void foo_decls(void);
void bar3() {
  foo_decls();
}
inline __attribute__((target("default"))) void foo_decls(void) {}
inline __attribute__((target("sse4.2"))) void foo_decls(void) {}

// CHECK: @foo.ifunc = ifunc i32 (), i32 ()* ()* @foo.resolver
// CHECK: @foo_inline.ifunc = ifunc i32 (), i32 ()* ()* @foo_inline.resolver
// CHECK: @foo_decls.ifunc = ifunc void (), void ()* ()* @foo_decls.resolver

// CHECK: define i32 @foo.sse4.2()
// CHECK: ret i32 0
// CHECK: define i32 @foo.arch_ivybridge()
// CHECK: ret i32 1
// CHECK: define i32 @foo()
// CHECK: ret i32 2
// CHECK: define i32 @bar()
// CHECK: call i32 @foo.ifunc()

// CHECK: define i32 ()* @foo.resolver() comdat
// CHECK: call void @__cpu_indicator_init()
// CHECK: ret i32 ()* @foo.arch_sandybridge
// CHECK: ret i32 ()* @foo.arch_ivybridge
// CHECK: ret i32 ()* @foo.sse4.2
// CHECK: ret i32 ()* @foo

// CHECK: define i32 @bar2()
// CHECK: call i32 @foo_inline.ifunc()

// CHECK: define i32 ()* @foo_inline.resolver() comdat
// CHECK: call void @__cpu_indicator_init()
// CHECK: ret i32 ()* @foo_inline.arch_sandybridge
// CHECK: ret i32 ()* @foo_inline.arch_ivybridge
// CHECK: ret i32 ()* @foo_inline.sse4.2
// CHECK: ret i32 ()* @foo_inline

// CHECK: define void @bar3()
// CHECK: call void @foo_decls.ifunc()

// CHECK: define void ()* @foo_decls.resolver() comdat
// CHECK: ret void ()* @foo_decls.sse4.2
// CHECK: ret void ()* @foo_decls

// CHECK: declare i32 @foo.arch_sandybridge()

// CHECK: define available_externally i32 @foo_inline.sse4.2()
// CHECK: ret i32 0

// CHECK: declare i32 @foo_inline.arch_sandybridge()
//
// CHECK: define available_externally i32 @foo_inline.arch_ivybridge()
// CHECK: ret i32 1
// CHECK: define available_externally i32 @foo_inline()
// CHECK: ret i32 2

// CHECK: define available_externally void @foo_decls()
// CHECK: define available_externally void @foo_decls.sse4.2()

