// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=ITANIUM,LINUX
// RUN: %clang_cc1 -triple x86_64-apple-macos -emit-llvm %s -o - | FileCheck %s --check-prefixes=ITANIUM,DARWIN
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
int __attribute__((target("arch=sierraforest"))) foo(void) {return 16;}
int __attribute__((target("arch=grandridge"))) foo(void) {return 17;}
int __attribute__((target("arch=graniterapids"))) foo(void) {return 18;}
int __attribute__((target("arch=emeraldrapids"))) foo(void) {return 19;}
int __attribute__((target("arch=graniterapids-d"))) foo(void) {return 20;}
int __attribute__((target("arch=arrowlake"))) foo(void) {return 21;}
int __attribute__((target("arch=arrowlake-s"))) foo(void) {return 22;}
int __attribute__((target("arch=lunarlake"))) foo(void) {return 23;}
int __attribute__((target("arch=gracemont"))) foo(void) {return 24;}
int __attribute__((target("arch=pantherlake"))) foo(void) {return 25;}
int __attribute__((target("arch=clearwaterforest"))) foo(void) {return 26;}
int __attribute__((target("arch=diamondrapids"))) foo(void) {return 27;}
int __attribute__((target("arch=wildcatlake"))) foo(void) {return 28;}
int __attribute__((target("default"))) foo(void) { return 2; }

int bar(void) {
  return foo();
}

static int __attribute__((target("arch=meteorlake"))) foo_internal(void) {return 15;}
static int __attribute__((target("default"))) foo_internal(void) { return 2; }

int bar1(void) {
  return foo_internal();
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

// DARWIN-NOT: comdat

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
// DARWIN: @llvm.used = appending global [2 x ptr] [ptr @foo_used, ptr @foo_used2.avx_sse4.2], section "llvm.metadata"
// WINDOWS: @llvm.used = appending global [2 x ptr] [ptr @foo_used, ptr @foo_used2.avx_sse4.2], section "llvm.metadata"


// ITANIUM: @foo.ifunc = weak_odr ifunc i32 (), ptr @foo.resolver
// ITANIUM: @foo_internal.ifunc = internal ifunc i32 (), ptr @foo_internal.resolver
// ITANIUM: @foo_inline.ifunc = weak_odr ifunc i32 (), ptr @foo_inline.resolver
// ITANIUM: @foo_decls.ifunc = weak_odr ifunc void (), ptr @foo_decls.resolver
// ITANIUM: @foo_multi.ifunc = weak_odr ifunc void (i32, double), ptr @foo_multi.resolver
// ITANIUM: @fwd_decl_default.ifunc = weak_odr ifunc i32 (), ptr @fwd_decl_default.resolver
// ITANIUM: @fwd_decl_avx.ifunc = weak_odr ifunc i32 (), ptr @fwd_decl_avx.resolver

// ITANIUM: define{{.*}} i32 @foo.sse4.2()
// ITANIUM: ret i32 0
// ITANIUM: define{{.*}} i32 @foo.arch_ivybridge()
// ITANIUM: ret i32 1
// ITANIUM: define{{.*}} i32 @foo.arch_goldmont()
// ITANIUM: ret i32 3
// ITANIUM: define{{.*}} i32 @foo.arch_goldmont-plus()
// ITANIUM: ret i32 4
// ITANIUM: define{{.*}} i32 @foo.arch_tremont()
// ITANIUM: ret i32 5
// ITANIUM: define{{.*}} i32 @foo.arch_icelake-client()
// ITANIUM: ret i32 6
// ITANIUM: define{{.*}} i32 @foo.arch_icelake-server()
// ITANIUM: ret i32 7
// ITANIUM: define{{.*}} i32 @foo.arch_cooperlake()
// ITANIUM: ret i32 8
// ITANIUM: define{{.*}} i32 @foo.arch_tigerlake()
// ITANIUM: ret i32 9
// ITANIUM: define{{.*}} i32 @foo.arch_sapphirerapids()
// ITANIUM: ret i32 10
// ITANIUM: define{{.*}} i32 @foo.arch_alderlake()
// ITANIUM: ret i32 11
// ITANIUM: define{{.*}} i32 @foo.arch_rocketlake()
// ITANIUM: ret i32 12
// ITANIUM: define{{.*}} i32 @foo.arch_core2()
// ITANIUM: ret i32 13
// ITANIUM: define{{.*}} i32 @foo.arch_raptorlake()
// ITANIUM: ret i32 14
// ITANIUM: define{{.*}} i32 @foo.arch_meteorlake()
// ITANIUM: ret i32 15
// ITANIUM: define{{.*}} i32 @foo.arch_sierraforest()
// ITANIUM: ret i32 16
// ITANIUM: define{{.*}} i32 @foo.arch_grandridge()
// ITANIUM: ret i32 17
// ITANIUM: define{{.*}} i32 @foo.arch_graniterapids()
// ITANIUM: ret i32 18
// ITANIUM: define{{.*}} i32 @foo.arch_emeraldrapids()
// ITANIUM: ret i32 19
// ITANIUM: define{{.*}} i32 @foo.arch_graniterapids-d()
// ITANIUM: ret i32 20
// ITANIUM: define{{.*}} i32 @foo.arch_arrowlake()
// ITANIUM: ret i32 21
// ITANIUM: define{{.*}} i32 @foo.arch_arrowlake-s()
// ITANIUM: ret i32 22
// ITANIUM: define{{.*}} i32 @foo.arch_lunarlake()
// ITANIUM: ret i32 23
// ITANIUM: define{{.*}} i32 @foo.arch_gracemont()
// ITANIUM: ret i32 24
// ITANIUM: define{{.*}} i32 @foo.arch_pantherlake()
// ITANIUM: ret i32 25
// ITANIUM: define{{.*}} i32 @foo.arch_clearwaterforest()
// ITANIUM: ret i32 26
// ITANIUM: define{{.*}} i32 @foo.arch_diamondrapids()
// ITANIUM: ret i32 27
// ITANIUM: define{{.*}} i32 @foo.arch_wildcatlake()
// ITANIUM: ret i32 28
// ITANIUM: define{{.*}} i32 @foo()
// ITANIUM: ret i32 2
// ITANIUM: define{{.*}} i32 @bar()
// ITANIUM: call i32 @foo.ifunc()

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
// WINDOWS: define{{.*}} i32 @foo.arch_sierraforest()
// WINDOWS: ret i32 16
// WINDOWS: define{{.*}} i32 @foo.arch_grandridge()
// WINDOWS: ret i32 17
// WINDOWS: define{{.*}} i32 @foo.arch_graniterapids()
// WINDOWS: ret i32 18
// WINDOWS: define dso_local i32 @foo.arch_emeraldrapids()
// WINDOWS: ret i32 19
// WINDOWS: define dso_local i32 @foo.arch_graniterapids-d()
// WINDOWS: ret i32 20
// WINDOWS: define dso_local i32 @foo.arch_arrowlake()
// WINDOWS: ret i32 21
// WINDOWS: define dso_local i32 @foo.arch_arrowlake-s()
// WINDOWS: ret i32 22
// WINDOWS: define dso_local i32 @foo.arch_lunarlake()
// WINDOWS: ret i32 23
// WINDOWS: define dso_local i32 @foo.arch_gracemont()
// WINDOWS: ret i32 24
// WINDOWS: define dso_local i32 @foo.arch_pantherlake()
// WINDOWS: ret i32 25
// WINDOWS: define dso_local i32 @foo.arch_clearwaterforest()
// WINDOWS: ret i32 26
// WINDOWS: define dso_local i32 @foo.arch_diamondrapids()
// WINDOWS: ret i32 27
// WINDOWS: define dso_local i32 @foo.arch_wildcatlake()
// WINDOWS: ret i32 28
// WINDOWS: define dso_local i32 @foo()
// WINDOWS: ret i32 2
// WINDOWS: define dso_local i32 @bar()
// WINDOWS: call i32 @foo.resolver()

// ITANIUM: define weak_odr ptr @foo.resolver()
// LINUX-SAME: comdat
// ITANIUM: call void @__cpu_indicator_init()
// ITANIUM: ret ptr @foo.arch_sandybridge
// ITANIUM: ret ptr @foo.arch_ivybridge
// ITANIUM: ret ptr @foo.sse4.2
// ITANIUM: ret ptr @foo

// WINDOWS: define weak_odr dso_local i32 @foo.resolver() comdat
// WINDOWS: call void @__cpu_indicator_init()
// WINDOWS: call i32 @foo.arch_sandybridge
// WINDOWS: call i32 @foo.arch_ivybridge
// WINDOWS: call i32 @foo.sse4.2
// WINDOWS: call i32 @foo

/// Internal linkage resolvers do not use comdat.
// ITANIUM: define internal ptr @foo_internal.resolver() {

// WINDOWS: define internal i32 @foo_internal.resolver() {

// ITANIUM: define{{.*}} i32 @bar2()
// ITANIUM: call i32 @foo_inline.ifunc()

// WINDOWS: define dso_local i32 @bar2()
// WINDOWS: call i32 @foo_inline.resolver()

// ITANIUM: define weak_odr ptr @foo_inline.resolver()
// LINUX-SAME: comdat
// ITANIUM: call void @__cpu_indicator_init()
// ITANIUM: ret ptr @foo_inline.arch_sandybridge
// ITANIUM: ret ptr @foo_inline.arch_ivybridge
// ITANIUM: ret ptr @foo_inline.sse4.2
// ITANIUM: ret ptr @foo_inline

// WINDOWS: define weak_odr dso_local i32 @foo_inline.resolver() comdat
// WINDOWS: call void @__cpu_indicator_init()
// WINDOWS: call i32 @foo_inline.arch_sandybridge
// WINDOWS: call i32 @foo_inline.arch_ivybridge
// WINDOWS: call i32 @foo_inline.sse4.2
// WINDOWS: call i32 @foo_inline

// ITANIUM: define{{.*}} void @bar3()
// ITANIUM: call void @foo_decls.ifunc()

// WINDOWS: define dso_local void @bar3()
// WINDOWS: call void @foo_decls.resolver()

// ITANIUM: define weak_odr ptr @foo_decls.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @foo_decls.sse4.2
// ITANIUM: ret ptr @foo_decls

// WINDOWS: define weak_odr dso_local void @foo_decls.resolver() comdat
// WINDOWS: call void @foo_decls.sse4.2
// WINDOWS: call void @foo_decls

// ITANIUM: define{{.*}} void @bar4()
// ITANIUM: call void @foo_multi.ifunc(i32 noundef 1, double noundef 5.{{[0+e]*}})

// WINDOWS: define dso_local void @bar4()
// WINDOWS: call void @foo_multi.resolver(i32 noundef 1, double noundef 5.{{[0+e]*}})

// ITANIUM: define weak_odr ptr @foo_multi.resolver()
// LINUX-SAME: comdat
// ITANIUM: and i32 %{{.*}}, 4352
// ITANIUM: icmp eq i32 %{{.*}}, 4352
// ITANIUM: ret ptr @foo_multi.fma4_sse4.2
// ITANIUM: icmp eq i32 %{{.*}}, 12
// ITANIUM: and i32 %{{.*}}, 4352
// ITANIUM: icmp eq i32 %{{.*}}, 4352
// ITANIUM: ret ptr @foo_multi.arch_ivybridge_fma4_sse4.2
// ITANIUM: and i32 %{{.*}}, 768
// ITANIUM: icmp eq i32 %{{.*}}, 768
// ITANIUM: ret ptr @foo_multi.avx_sse4.2
// ITANIUM: ret ptr @foo_multi

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

// ITANIUM: define{{.*}} i32 @fwd_decl_default()
// ITANIUM: ret i32 2
// ITANIUM: define{{.*}} i32 @fwd_decl_avx.avx()
// ITANIUM: ret i32 2
// ITANIUM: define{{.*}} i32 @fwd_decl_avx()
// ITANIUM: ret i32 2

// WINDOWS: define dso_local i32 @fwd_decl_default()
// WINDOWS: ret i32 2
// WINDOWS: define dso_local i32 @fwd_decl_avx.avx()
// WINDOWS: ret i32 2
// WINDOWS: define dso_local i32 @fwd_decl_avx()
// WINDOWS: ret i32 2

// ITANIUM: define{{.*}} void @bar5()
// ITANIUM: call i32 @fwd_decl_default.ifunc()
// ITANIUM: call i32 @fwd_decl_avx.ifunc()

// WINDOWS: define dso_local void @bar5()
// WINDOWS: call i32 @fwd_decl_default.resolver()
// WINDOWS: call i32 @fwd_decl_avx.resolver()

// ITANIUM: define weak_odr ptr @fwd_decl_default.resolver()
// LINUX-SAME: comdat
// ITANIUM: call void @__cpu_indicator_init()
// ITANIUM: ret ptr @fwd_decl_default
// ITANIUM: define weak_odr ptr @fwd_decl_avx.resolver()
// LINUX-SAME: comdat
// ITANIUM: call void @__cpu_indicator_init()
// ITANIUM: ret ptr @fwd_decl_avx.avx
// ITANIUM: ret ptr @fwd_decl_avx

// WINDOWS: define weak_odr dso_local i32 @fwd_decl_default.resolver() comdat
// WINDOWS: call void @__cpu_indicator_init()
// WINDOWS: call i32 @fwd_decl_default
// WINDOWS: define weak_odr dso_local i32 @fwd_decl_avx.resolver() comdat
// WINDOWS: call void @__cpu_indicator_init()
// WINDOWS: call i32 @fwd_decl_avx.avx
// WINDOWS: call i32 @fwd_decl_avx

// ITANIUM: define{{.*}} i32 @changed_to_mv.avx()
// ITANIUM: define{{.*}} i32 @changed_to_mv.fma4()

// WINDOWS: define dso_local i32 @changed_to_mv.avx()
// WINDOWS: define dso_local i32 @changed_to_mv.fma4()

// ITANIUM: define linkonce void @foo_used(i32 noundef %{{.*}}, double noundef %{{.*}})
// ITANIUM-NOT: @foo_used.avx_sse4.2(
// ITANIUM-NOT: @foo_used2(
// ITANIUM: define linkonce void @foo_used2.avx_sse4.2(i32 noundef %{{.*}}, double noundef %{{.*}})

// WINDOWS: define linkonce_odr dso_local void @foo_used(i32 noundef %{{.*}}, double noundef %{{.*}})
// WINDOWS-NOT: @foo_used.avx_sse4.2(
// WINDOWS-NOT: @foo_used2(
// WINDOWS: define linkonce_odr dso_local void @foo_used2.avx_sse4.2(i32 noundef %{{.*}}, double noundef %{{.*}})

// ITANIUM: declare i32 @foo.arch_sandybridge()
// WINDOWS: declare dso_local i32 @foo.arch_sandybridge()

// ITANIUM: define linkonce i32 @foo_inline.sse4.2()
// ITANIUM: ret i32 0

// WINDOWS: define linkonce_odr dso_local i32 @foo_inline.sse4.2()
// WINDOWS: ret i32 0

// ITANIUM: declare i32 @foo_inline.arch_sandybridge()

// WINDOWS: declare dso_local i32 @foo_inline.arch_sandybridge()

// ITANIUM: define linkonce i32 @foo_inline.arch_ivybridge()
// ITANIUM: ret i32 1
// ITANIUM: define linkonce i32 @foo_inline()
// ITANIUM: ret i32 2

// WINDOWS: define linkonce_odr dso_local i32 @foo_inline.arch_ivybridge()
// WINDOWS: ret i32 1
// WINDOWS: define linkonce_odr dso_local i32 @foo_inline()
// WINDOWS: ret i32 2

// ITANIUM: define linkonce void @foo_decls()
// ITANIUM: define linkonce void @foo_decls.sse4.2()

// WINDOWS: define linkonce_odr dso_local void @foo_decls()
// WINDOWS: define linkonce_odr dso_local void @foo_decls.sse4.2()

// ITANIUM: define linkonce void @foo_multi(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})
// ITANIUM: define linkonce void @foo_multi.avx_sse4.2(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})
// ITANIUM: define linkonce void @foo_multi.fma4_sse4.2(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})
// ITANIUM: define linkonce void @foo_multi.arch_ivybridge_fma4_sse4.2(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})

// WINDOWS: define linkonce_odr dso_local void @foo_multi(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})
// WINDOWS: define linkonce_odr dso_local void @foo_multi.avx_sse4.2(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})
// WINDOWS: define linkonce_odr dso_local void @foo_multi.fma4_sse4.2(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})
// WINDOWS: define linkonce_odr dso_local void @foo_multi.arch_ivybridge_fma4_sse4.2(i32 noundef %{{[^,]+}}, double noundef %{{[^\)]+}})

// Ensure that we emit the 'static' function here.
// ITANIUM: define linkonce void @pr50025()
// ITANIUM: call void @must_be_emitted
// ITANIUM: define internal void @must_be_emitted()
// WINDOWS: define linkonce_odr dso_local void @pr50025() #{{[0-9]*}} comdat
// WINDOWS: call void @must_be_emitted
// WINDOWS: define internal void @must_be_emitted()

// ITANIUM: define linkonce void @pr50025c()
// ITANIUM: call void @pr50025b.ifunc()
// WINDOWS: define linkonce_odr dso_local void @pr50025c() #{{[0-9]*}} comdat
// WINDOWS: call void @pr50025b.resolver()

// ITANIUM: define weak_odr ptr @pr50025b.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @pr50025b
// ITANIUM: define linkonce void @pr50025b()
// ITANIUM: call void @must_be_emitted()
// WINDOWS: define weak_odr dso_local void @pr50025b.resolver() comdat
// WINDOWS: musttail call void @pr50025b()
// WINDOWS: define linkonce_odr dso_local void @pr50025b() #{{[0-9]*}} comdat
// WINDOWS: call void @must_be_emitted()
