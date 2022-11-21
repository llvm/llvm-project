// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -triple x86_64-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS

int __attribute__((target("sse4.2"))) foo(void) { return 0; }
int __attribute__((target("arch=sandybridge"))) foo(void);
int __attribute__((target("arch=ivybridge"))) foo(void) {return 1;}
int __attribute__((target("arch=goldmont"))) foo(void) {return 3;}
int __attribute__((target("arch=goldmont-plus"))) foo(void) {return 4;}
int __attribute__((target("arch=tremont"))) foo(void) {return 5;}
int __attribute__((target("arch=icelake-client"))) foo(void) {return 6;}
int __attribute__((target("arch=icelake-server"))) foo(void) {return 7;}
int __attribute__((target("arch=cooperlake"))) foo(void) {return 8;}
int __attribute__((target("arch=tigerlake"))) foo(void) {return 9;}
int __attribute__((target("arch=sapphirerapids"))) foo(void) {return 10;}
int __attribute__((target("arch=alderlake"))) foo(void) {return 11;}
int __attribute__((target("arch=rocketlake"))) foo(void) {return 12;}
int __attribute__((target("arch=core2"))) foo(void) {return 13;}
int __attribute__((target("arch=raptorlake"))) foo(void) {return 14;}
int __attribute__((target("arch=meteorlake"))) foo(void) {return 15;}
int __attribute__((target("default"))) foo(void) { return 2; }

int bar(void) {
  return foo();
}

inline int __attribute__((target("sse4.2"))) foo_inline(void) { return 0; }
inline int __attribute__((target("arch=sandybridge"))) foo_inline(void);
inline int __attribute__((target("arch=ivybridge"))) foo_inline(void) {return 1;}
inline int __attribute__((target("default"))) foo_inline(void) { return 2; }

int bar2(void) {
  return foo_inline();
}

inline __attribute__((target("default"))) void foo_decls(void);
inline __attribute__((target("sse4.2"))) void foo_decls(void);
void bar3(void) {
  foo_decls();
}
inline __attribute__((target("default"))) void foo_decls(void) {}
inline __attribute__((target("sse4.2"))) void foo_decls(void) {}

inline __attribute__((target("default"))) void foo_multi(int i, double d) {}
inline __attribute__((target("avx,sse4.2"))) void foo_multi(int i, double d) {}
inline __attribute__((target("sse4.2,fma4"))) void foo_multi(int i, double d) {}
inline __attribute__((target("arch=ivybridge,fma4,sse4.2"))) void foo_multi(int i, double d) {}
void bar4(void) {
  foo_multi(1, 5.0);
}

int fwd_decl_default(void);
int __attribute__((target("default"))) fwd_decl_default(void) { return 2; }

int fwd_decl_avx(void);
int __attribute__((target("avx"))) fwd_decl_avx(void) { return 2; }
int __attribute__((target("default"))) fwd_decl_avx(void) { return 2; }

void bar5(void) {
  fwd_decl_default();
  fwd_decl_avx();
}

int __attribute__((target("avx"))) changed_to_mv(void) { return 0;}
int __attribute__((target("fma4"))) changed_to_mv(void) { return 1;}

__attribute__((target("default"), used)) inline void foo_used(int i, double d) {}
__attribute__((target("avx,sse4.2"))) inline void foo_used(int i, double d) {}

__attribute__((target("default"))) inline void foo_used2(int i, double d) {}
__attribute__((target("avx,sse4.2"), used)) inline void foo_used2(int i, double d) {}

// PR50025:
static void must_be_emitted(void) {}
inline __attribute__((target("default"))) void pr50025(void) { must_be_emitted(); }
void calls_pr50025(void) { pr50025(); }

// Also need to make sure we get other multiversion functions.
inline __attribute__((target("default"))) void pr50025b(void) { must_be_emitted(); }
inline __attribute__((target("default"))) void pr50025c(void) { pr50025b(); }
void calls_pr50025c(void) { pr50025c(); }

// LINUX: $foo.resolver = comdat any
// LINUX: $foo_inline.resolver = comdat any
// LINUX: $foo_decls.resolver = comdat any
// LINUX: $foo_multi.resolver = comdat any
// LINUX: $fwd_decl_default.resolver = comdat any
// LINUX: $fwd_decl_avx.resolver = comdat any
// LINUX: $pr50025.resolver = comdat any
// LINUX: $pr50025c.resolver = comdat any
// LINUX: $pr50025b.resolver = comdat any

// WINDOWS: $foo.resolver = comdat any
// WINDOWS: $foo_inline.resolver = comdat any
// WINDOWS: $foo_decls.resolver = comdat any
// WINDOWS: $foo_multi.resolver = comdat any
// WINDOWS: $fwd_decl_default.resolver = comdat any
// WINDOWS: $fwd_decl_avx.resolver = comdat any
// WINDOWS: $foo_used = comdat any
// WINDOWS: $foo_used2.avx_sse4.2 = comdat any
// WINDOWS: $pr50025.resolver = comdat any
// WINDOWS: $pr50025c.resolver = comdat any
// WINDOWS: $foo_inline.sse4.2 = comdat any
// WINDOWS: $foo_inline.arch_ivybridge = comdat any
// WINDOWS: $foo_inline = comdat any
// WINDOWS: $foo_decls = comdat any
// WINDOWS: $foo_decls.sse4.2 = comdat any
// WINDOWS: $foo_multi = comdat any
// WINDOWS: $foo_multi.avx_sse4.2 = comdat any
// WINDOWS: $foo_multi.fma4_sse4.2 = comdat any
// WINDOWS: $foo_multi.arch_ivybridge_fma4_sse4.2 = comdat any
// WINDOWS: $pr50025 = comdat any
// WINDOWS: $pr50025c = comdat any
// WINDOWS: $pr50025b.resolver = comdat any
// WINDOWS: $pr50025b = comdat any


// LINUX: @llvm.compiler.used = appending global [2 x ptr] [ptr @foo_used, ptr @foo_used2.avx_sse4.2], section "llvm.metadata"
// WINDOWS: @llvm.used = appending global [2 x ptr] [ptr @foo_used, ptr @foo_used2.avx_sse4.2], section "llvm.metadata"


// LINUX: @foo.ifunc = weak_odr ifunc i32 (), ptr @foo.resolver
// LINUX: @foo_inline.ifunc = weak_odr ifunc i32 (), ptr @foo_inline.resolver
// LINUX: @foo_decls.ifunc = weak_odr ifunc void (), ptr @foo_decls.resolver
// LINUX: @foo_multi.ifunc = weak_odr ifunc void (i32, double), ptr @foo_multi.resolver
// LINUX: @fwd_decl_default.ifunc = weak_odr ifunc i32 (), ptr @fwd_decl_default.resolver
// LINUX: @fwd_decl_avx.ifunc = weak_odr ifunc i32 (), ptr @fwd_decl_avx.resolver

// LINUX: define{{.*}} i32 @foo.sse4.2()
// LINUX: ret i32 0
// LINUX: define{{.*}} i32 @foo.arch_ivybridge()
// LINUX: ret i32 1
// LINUX: define{{.*}} i32 @foo.arch_goldmont()
// LINUX: ret i32 3
// LINUX: define{{.*}} i32 @foo.arch_goldmont-plus()
// LINUX: ret i32 4
// LINUX: define{{.*}} i32 @foo.arch_tremont()
// LINUX: ret i32 5
// LINUX: define{{.*}} i32 @foo.arch_icelake-client()
// LINUX: ret i32 6
// LINUX: define{{.*}} i32 @foo.arch_icelake-server()
// LINUX: ret i32 7
// LINUX: define{{.*}} i32 @foo.arch_cooperlake()
// LINUX: ret i32 8
// LINUX: define{{.*}} i32 @foo.arch_tigerlake()
// LINUX: ret i32 9
// LINUX: define{{.*}} i32 @foo.arch_sapphirerapids()
// LINUX: ret i32 10
// LINUX: define{{.*}} i32 @foo.arch_alderlake()
// LINUX: ret i32 11
// LINUX: define{{.*}} i32 @foo.arch_rocketlake()
// LINUX: ret i32 12
// LINUX: define{{.*}} i32 @foo.arch_core2()
// LINUX: ret i32 13
// LINUX: define{{.*}} i32 @foo.arch_raptorlake()
// LINUX: ret i32 14
// LINUX: define{{.*}} i32 @foo.arch_meteorlake()
// LINUX: ret i32 15
// LINUX: define{{.*}} i32 @foo()
// LINUX: ret i32 2
// LINUX: define{{.*}} i32 @bar()
// LINUX: call i32 @foo.ifunc()

// WINDOWS: define dso_local i32 @foo.sse4.2()
// WINDOWS: ret i32 0
// WINDOWS: define dso_local i32 @foo.arch_ivybridge()
// WINDOWS: ret i32 1
// WINDOWS: define dso_local i32 @foo.arch_goldmont()
// WINDOWS: ret i32 3
// WINDOWS: define dso_local i32 @foo.arch_goldmont-plus()
// WINDOWS: ret i32 4
// WINDOWS: define dso_local i32 @foo.arch_tremont()
// WINDOWS: ret i32 5
// WINDOWS: define dso_local i32 @foo.arch_icelake-client()
// WINDOWS: ret i32 6
// WINDOWS: define dso_local i32 @foo.arch_icelake-server()
// WINDOWS: ret i32 7
// WINDOWS: define dso_local i32 @foo.arch_cooperlake()
// WINDOWS: ret i32 8
// WINDOWS: define dso_local i32 @foo.arch_tigerlake()
// WINDOWS: ret i32 9
// WINDOWS: define dso_local i32 @foo.arch_sapphirerapids()
// WINDOWS: ret i32 10
// WINDOWS: define dso_local i32 @foo.arch_alderlake()
// WINDOWS: ret i32 11
// WINDOWS: define dso_local i32 @foo.arch_rocketlake()
// WINDOWS: ret i32 12
// WINDOWS: define dso_local i32 @foo.arch_core2()
// WINDOWS: ret i32 13
// WINDOWS: define dso_local i32 @foo.arch_raptorlake()
// WINDOWS: ret i32 14
// WINDOWS: define dso_local i32 @foo.arch_meteorlake()
// WINDOWS: ret i32 15
// WINDOWS: define dso_local i32 @foo()
// WINDOWS: ret i32 2
// WINDOWS: define dso_local i32 @bar()
// WINDOWS: call i32 @foo.resolver()

// LINUX: define weak_odr ptr @foo.resolver() comdat
// LINUX: call void @__cpu_indicator_init()
// LINUX: ret ptr @foo.arch_sandybridge
// LINUX: ret ptr @foo.arch_ivybridge
// LINUX: ret ptr @foo.sse4.2
// LINUX: ret ptr @foo

// WINDOWS: define weak_odr dso_local i32 @foo.resolver() comdat
// WINDOWS: call void @__cpu_indicator_init()
// WINDOWS: call i32 @foo.arch_sandybridge
// WINDOWS: call i32 @foo.arch_ivybridge
// WINDOWS: call i32 @foo.sse4.2
// WINDOWS: call i32 @foo

// LINUX: define{{.*}} i32 @bar2()
// LINUX: call i32 @foo_inline.ifunc()

// WINDOWS: define dso_local i32 @bar2()
// WINDOWS: call i32 @foo_inline.resolver()

// LINUX: define weak_odr ptr @foo_inline.resolver() comdat
// LINUX: call void @__cpu_indicator_init()
// LINUX: ret ptr @foo_inline.arch_sandybridge
// LINUX: ret ptr @foo_inline.arch_ivybridge
// LINUX: ret ptr @foo_inline.sse4.2
// LINUX: ret ptr @foo_inline

// WINDOWS: define weak_odr dso_local i32 @foo_inline.resolver() comdat
// WINDOWS: call void @__cpu_indicator_init()
// WINDOWS: call i32 @foo_inline.arch_sandybridge
// WINDOWS: call i32 @foo_inline.arch_ivybridge
// WINDOWS: call i32 @foo_inline.sse4.2
// WINDOWS: call i32 @foo_inline

// LINUX: define{{.*}} void @bar3()
// LINUX: call void @foo_decls.ifunc()

// WINDOWS: define dso_local void @bar3()
// WINDOWS: call void @foo_decls.resolver()

// LINUX: define weak_odr ptr @foo_decls.resolver() comdat
// LINUX: ret ptr @foo_decls.sse4.2
// LINUX: ret ptr @foo_decls

// WINDOWS: define weak_odr dso_local void @foo_decls.resolver() comdat
// WINDOWS: call void @foo_decls.sse4.2
// WINDOWS: call void @foo_decls

// LINUX: define{{.*}} void @bar4()
// LINUX: call void @foo_multi.ifunc(i32 noundef 1, double noundef 5.{{[0+e]*}})

// WINDOWS: define dso_local void @bar4()
// WINDOWS: call void @foo_multi.resolver(i32 noundef 1, double noundef 5.{{[0+e]*}})

// LINUX: define weak_odr ptr @foo_multi.resolver() comdat
// LINUX: and i32 %{{.*}}, 4352
// LINUX: icmp eq i32 %{{.*}}, 4352
// LINUX: ret ptr @foo_multi.fma4_sse4.2
// LINUX: icmp eq i32 %{{.*}}, 12
// LINUX: and i32 %{{.*}}, 4352
// LINUX: icmp eq i32 %{{.*}}, 4352
// LINUX: ret ptr @foo_multi.arch_ivybridge_fma4_sse4.2
// LINUX: and i32 %{{.*}}, 768
// LINUX: icmp eq i32 %{{.*}}, 768
// LINUX: ret ptr @foo_multi.avx_sse4.2
// LINUX: ret ptr @foo_multi

// WINDOWS: define weak_odr dso_local void @foo_multi.resolver(i32 %0, double %1) comdat
// WINDOWS: and i32 %{{.*}}, 4352
// WINDOWS: icmp eq i32 %{{.*}}, 4352
// WINDOWS: call void @foo_multi.fma4_sse4.2(i32 %0, double %1)
// WINDOWS-NEXT: ret void
// WINDOWS: icmp eq i32 %{{.*}}, 12
// WINDOWS: and i32 %{{.*}}, 4352
// WINDOWS: icmp eq i32 %{{.*}}, 4352
// WINDOWS: call void @foo_multi.arch_ivybridge_fma4_sse4.2(i32 %0, double %1)
// WINDOWS-NEXT: ret void
// WINDOWS: and i32 %{{.*}}, 768
// WINDOWS: icmp eq i32 %{{.*}}, 768
// WINDOWS: call void @foo_multi.avx_sse4.2(i32 %0, double %1)
// WINDOWS-NEXT: ret void
// WINDOWS: call void @foo_multi(i32 %0, double %1)
// WINDOWS-NEXT: ret void

// LINUX: define{{.*}} i32 @fwd_decl_default()
// LINUX: ret i32 2
// LINUX: define{{.*}} i32 @fwd_decl_avx.avx()
// LINUX: ret i32 2
// LINUX: define{{.*}} i32 @fwd_decl_avx()
// LINUX: ret i32 2

// WINDOWS: define dso_local i32 @fwd_decl_default()
// WINDOWS: ret i32 2
// WINDOWS: define dso_local i32 @fwd_decl_avx.avx()
// WINDOWS: ret i32 2
// WINDOWS: define dso_local i32 @fwd_decl_avx()
// WINDOWS: ret i32 2

// LINUX: define{{.*}} void @bar5()
// LINUX: call i32 @fwd_decl_default.ifunc()
// LINUX: call i32 @fwd_decl_avx.ifunc()

// WINDOWS: define dso_local void @bar5()
// WINDOWS: call i32 @fwd_decl_default.resolver()
// WINDOWS: call i32 @fwd_decl_avx.resolver()

// LINUX: define weak_odr ptr @fwd_decl_default.resolver() comdat
// LINUX: call void @__cpu_indicator_init()
// LINUX: ret ptr @fwd_decl_default
// LINUX: define weak_odr ptr @fwd_decl_avx.resolver() comdat
// LINUX: call void @__cpu_indicator_init()
// LINUX: ret ptr @fwd_decl_avx.avx
// LINUX: ret ptr @fwd_decl_avx

// WINDOWS: define weak_odr dso_local i32 @fwd_decl_default.resolver() comdat
// WINDOWS: call void @__cpu_indicator_init()
// WINDOWS: call i32 @fwd_decl_default
// WINDOWS: define weak_odr dso_local i32 @fwd_decl_avx.resolver() comdat
// WINDOWS: call void @__cpu_indicator_init()
// WINDOWS: call i32 @fwd_decl_avx.avx
// WINDOWS: call i32 @fwd_decl_avx

// LINUX: define{{.*}} i32 @changed_to_mv.avx()
// LINUX: define{{.*}} i32 @changed_to_mv.fma4()

// WINDOWS: define dso_local i32 @changed_to_mv.avx()
// WINDOWS: define dso_local i32 @changed_to_mv.fma4()

// LINUX: define linkonce void @foo_used(i32 noundef %{{.*}}, double noundef %{{.*}})
// LINUX-NOT: @foo_used.avx_sse4.2(
// LINUX-NOT: @foo_used2(
// LINUX: define linkonce void @foo_used2.avx_sse4.2(i32 noundef %{{.*}}, double noundef %{{.*}})

// WINDOWS: define linkonce_odr dso_local void @foo_used(i32 noundef %{{.*}}, double noundef %{{.*}})
// WINDOWS-NOT: @foo_used.avx_sse4.2(
// WINDOWS-NOT: @foo_used2(
// WINDOWS: define linkonce_odr dso_local void @foo_used2.avx_sse4.2(i32 noundef %{{.*}}, double noundef %{{.*}})

// LINUX: declare i32 @foo.arch_sandybridge()
// WINDOWS: declare dso_local i32 @foo.arch_sandybridge()

// LINUX: define linkonce i32 @foo_inline.sse4.2()
// LINUX: ret i32 0

// WINDOWS: define linkonce_odr dso_local i32 @foo_inline.sse4.2()
// WINDOWS: ret i32 0

// LINUX: declare i32 @foo_inline.arch_sandybridge()

// WINDOWS: declare dso_local i32 @foo_inline.arch_sandybridge()

// LINUX: define linkonce i32 @foo_inline.arch_ivybridge()
// LINUX: ret i32 1
// LINUX: define linkonce i32 @foo_inline()
// LINUX: ret i32 2

// WINDOWS: define linkonce_odr dso_local i32 @foo_inline.arch_ivybridge()
// WINDOWS: ret i32 1
// WINDOWS: define linkonce_odr dso_local i32 @foo_inline()
// WINDOWS: ret i32 2

// LINUX: define linkonce void @foo_decls()
// LINUX: define linkonce void @foo_decls.sse4.2()

// WINDOWS: define linkonce_odr dso_local void @foo_decls()
// WINDOWS: define linkonce_odr dso_local void @foo_decls.sse4.2()

// LINUX: define linkonce void @foo_multi(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})
// LINUX: define linkonce void @foo_multi.avx_sse4.2(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})
// LINUX: define linkonce void @foo_multi.fma4_sse4.2(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})
// LINUX: define linkonce void @foo_multi.arch_ivybridge_fma4_sse4.2(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})

// WINDOWS: define linkonce_odr dso_local void @foo_multi(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})
// WINDOWS: define linkonce_odr dso_local void @foo_multi.avx_sse4.2(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})
// WINDOWS: define linkonce_odr dso_local void @foo_multi.fma4_sse4.2(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})
// WINDOWS: define linkonce_odr dso_local void @foo_multi.arch_ivybridge_fma4_sse4.2(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})

// Ensure that we emit the 'static' function here.
// LINUX: define linkonce void @pr50025()
// LINUX: call void @must_be_emitted
// LINUX: define internal void @must_be_emitted()
// WINDOWS: define linkonce_odr dso_local void @pr50025() #{{[0-9]*}} comdat
// WINDOWS: call void @must_be_emitted
// WINDOWS: define internal void @must_be_emitted()

// LINUX: define linkonce void @pr50025c()
// LINUX: call void @pr50025b.ifunc()
// WINDOWS: define linkonce_odr dso_local void @pr50025c() #{{[0-9]*}} comdat
// WINDOWS: call void @pr50025b.resolver()

// LINUX: define weak_odr ptr @pr50025b.resolver() comdat
// LINUX: ret ptr @pr50025b
// LINUX: define linkonce void @pr50025b()
// LINUX: call void @must_be_emitted()
// WINDOWS: define weak_odr dso_local void @pr50025b.resolver() comdat
// WINDOWS: musttail call void @pr50025b()
// WINDOWS: define linkonce_odr dso_local void @pr50025b() #{{[0-9]*}} comdat
// WINDOWS: call void @must_be_emitted()
