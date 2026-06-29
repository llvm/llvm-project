// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -target-cpu pwr7 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -target-cpu pwr7 -emit-llvm %s -o - | FileCheck %s

// CHECK: @internal = internal ifunc i32 (), ptr @internal.resolver
// CHECK: @foo = ifunc i32 (), ptr @foo.resolver
// CHECK: @foo_dupes = ifunc void (), ptr @foo_dupes.resolver
// CHECK: @unused = ifunc void (), ptr @unused.resolver
// CHECK: @foo_inline = linkonce ifunc i32 (), ptr @foo_inline.resolver
// CHECK: @foo_ref_then_def = ifunc i32 (), ptr @foo_ref_then_def.resolver
// CHECK: @foo_priority = ifunc i32 (i32), ptr @foo_priority.resolver
// CHEECK: @isa_level = ifunc i32 (i32), ptr @isa_level.resolver


static int __attribute__((target_clones("cpu=power10, default"))) internal(void) { return 0; }
int use(void) { return internal(); }
// CHECK: define internal ptr @internal.resolver() 

// test all supported cpus
int __attribute__((target_clones("cpu=power10, cpu=power11, cpu=pwr9, cpu=pwr7, cpu=power8, default"))) foo(void) { return 0; }
// CHECK: define internal {{.*}}i32 @foo.cpu_pwr10() #[[#ATTR_P10:]]
// CHECK: define internal {{.*}}i32 @foo.cpu_pwr11() #[[#ATTR_P11:]]
// CHECK: define internal {{.*}}i32 @foo.cpu_pwr9() #[[#ATTR_P9:]]
// CHECK: define internal {{.*}}i32 @foo.cpu_pwr7() #[[#ATTR_P7:]]
// CHECK: define internal {{.*}}i32 @foo.cpu_pwr8() #[[#ATTR_P8:]]
// CHECK: define internal {{.*}}i32 @foo.default() #[[#ATTR_P7:]]
// CHECK: define internal ptr @foo.resolver()
// CHECK: ret ptr @foo.cpu_pwr11
// CHECK: ret ptr @foo.cpu_pwr10
// CHECK: ret ptr @foo.cpu_pwr9
// CHECK: ret ptr @foo.cpu_pwr8
// CHECK: ret ptr @foo.cpu_pwr7
// CHECK: ret ptr @foo.default

__attribute__((target_clones("default,default ,cpu=pwr8"))) void foo_dupes(void) {}
// CHECK: define internal void @foo_dupes.default() #[[#ATTR_P7]]
// CHECK: define internal void @foo_dupes.cpu_pwr8() #[[#ATTR_P8:]]
// CHECK: define internal ptr @foo_dupes.resolver()
// CHECK: ret ptr @foo_dupes.cpu_pwr8
// CHECK: ret ptr @foo_dupes.default

void bar2(void) {
  // CHECK: define {{.*}}void @bar2()
  foo_dupes();
  // CHECK: call void @foo_dupes()
}

int bar(void) {
  // CHECK: define {{.*}}i32 @bar()
  return foo();
  // CHECK: call {{.*}}i32 @foo()
}

void __attribute__((target_clones("default, cpu=pwr9"))) unused(void) {}
// CHECK: define internal void @unused.default() #[[#ATTR_P7]]
// CHECK: define internal void @unused.cpu_pwr9() #[[#ATTR_P9:]]
// CHECK: define internal ptr @unused.resolver()
// CHECK: ret ptr @unused.cpu_pwr9
// CHECK: ret ptr @unused.default

int __attribute__((target_clones("cpu=power10, default"))) inherited(void);
int inherited(void) { return 0; }
// CHECK: define internal {{.*}}i32 @inherited.cpu_pwr10() #[[#ATTR_P10]]
// CHECK: define internal {{.*}}i32 @inherited.default() #[[#ATTR_P7]]
// CHECK: define internal ptr @inherited.resolver()
// CHECK: ret ptr @inherited.cpu_pwr10
// CHECK: ret ptr @inherited.default


int test_inherited(void) {
  // CHECK: define {{.*}}i32 @test_inherited()
  return inherited();
  // CHECK: call {{.*}}i32 @inherited()
}

inline int __attribute__((target_clones("default,cpu=pwr8")))
foo_inline(void) { return 0; }
int __attribute__((target_clones("cpu=pwr7,default")))
foo_ref_then_def(void);

int bar3(void) {
  // CHECK: define {{.*}}i32 @bar3()
  return foo_inline() + foo_ref_then_def();
  // CHECK: call {{.*}}i32 @foo_inline()
  // CHECK: call {{.*}}i32 @foo_ref_then_def()
}

// CHECK: define internal ptr @foo_inline.resolver()
// CHECK: ret ptr @foo_inline.cpu_pwr8
// CHECK: ret ptr @foo_inline.default

int __attribute__((target_clones("cpu=pwr7,default")))
foo_ref_then_def(void){ return 0; }
// CHECK: define internal ptr @foo_ref_then_def.resolver()
// CHECK: ret ptr @foo_ref_then_def.cpu_pwr7
// CHECK: ret ptr @foo_ref_then_def.default

int __attribute__((target_clones("default", "cpu=pwr8")))
foo_unused_no_defn(void);
// CHECK-NOT: foo_unused_no_defn

int __attribute__((target_clones("default", "cpu=pwr9")))
foo_used_no_defn(void);

int test_foo_used_no_defn(void) {
  // CHECK: define {{.*}}i32 @test_foo_used_no_defn()
  return foo_used_no_defn();
  // CHECK: call {{.*}}i32 @foo_used_no_defn()
}
// CHECK: declare {{.*}}i32 @foo_used_no_defn()

// Test that the CPU conditions are checked from the most to the least
// restrictive (highest to lowest CPU). Also test the codegen for the
// conditions
int __attribute__((target_clones("cpu=pwr10", "cpu=pwr7", "cpu=pwr9", "default", "cpu=pwr8")))
foo_priority(int x) { return x & (x - 1); }
// CHECK: define internal ptr @foo_priority.resolver()
// CHECK-NEXT: entry
//   if (__builtin_cpu_supports("arch_3_1")) return &foo_priority.cpu_pwr10;
// CHECK-NEXT: %[[#L1:]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @_system_configuration, {{i32|i64}} 4)
// CHECK-NEXT: icmp uge i32 %[[#L1]], 262144
// CHECK: ret ptr @foo_priority.cpu_pwr10
//   if (__builtin_cpu_supports("arch_3_00")) return &foo_priority.cpu_pwr9;
// CHECK: %[[#L2:]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @_system_configuration, {{i32|i64}} 4)
// CHECK-NEXT: icmp uge i32 %[[#L2]], 131072
// CHECK: ret ptr @foo_priority.cpu_pwr9
//   if (__builtin_cpu_supports("arch_2_07")) return &foo_priority.cpu_pwr8;
// CHECK: %[[#L3:]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @_system_configuration, {{i32|i64}} 4)
// CHECK-NEXT: icmp uge i32 %[[#L3]], 65536
// CHECK: ret ptr @foo_priority.cpu_pwr8
//   if (__builtin_cpu_supports("arch_2_06")) return &foo_priority.cpu_pwr8;
// CHECK: %[[#L4:]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @_system_configuration, {{i32|i64}} 4)
// CHECK-NEXT: icmp uge i32 %[[#L4]], 32768
// CHECK: ret ptr @foo_priority.cpu_pwr7
// CHECK: ret ptr @foo_priority.default


// Test feature-based target_clones
int __attribute__((target_clones("altivec", "default")))
foo_altivec(void) { return 0; }
// CHECK: define internal {{.*}}i32 @foo_altivec.altivec()
// CHECK: define internal {{.*}}i32 @foo_altivec.default()
// CHECK: define internal ptr @foo_altivec.resolver()
//   if (__builtin_cpu_supports("altivec")) return &foo_altivec.altivec;
// CHECK: %[[#ALTIVEC:]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @_system_configuration, {{i32|i64}} 204)
// CHECK-NEXT: icmp ugt i32 %[[#ALTIVEC]], 0
// CHECK: ret ptr @foo_altivec.altivec
// CHECK: ret ptr @foo_altivec.default

int __attribute__((target_clones("vsx", "default")))
foo_vsx(void) { return 0; }
// CHECK: define internal {{.*}}i32 @foo_vsx.vsx()
// CHECK: define internal {{.*}}i32 @foo_vsx.default()
// CHECK: define internal ptr @foo_vsx.resolver()
//   if (__builtin_cpu_supports("vsx")) return &foo_vsx.vsx;
// CHECK: %[[#VSX:]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @_system_configuration, {{i32|i64}} 204)
// CHECK-NEXT: icmp ugt i32 %[[#VSX]], 1
// CHECK: ret ptr @foo_vsx.vsx
// CHECK: ret ptr @foo_vsx.default

int __attribute__((target_clones("htm", "default")))
foo_htm(void) { return 0; }
// CHECK: define internal {{.*}}i32 @foo_htm.htm()
// CHECK: define internal {{.*}}i32 @foo_htm.default()
// CHECK: define internal ptr @foo_htm.resolver()
//   if (__builtin_cpu_supports("htm")) return &foo_htm.htm;
// CHECK: %[[#HTM:]] = call i64 @getsystemcfg(i32 59)
// CHECK-NEXT: icmp ugt i64 %[[#HTM]], 0
// CHECK: ret ptr @foo_htm.htm
// CHECK: ret ptr @foo_htm.default



// Test multiple features with priority ordering (random source order)
// Source order: altivec, power8-vector, cpu=pwr11, vsx, power9-vector, default
// Resolver order by priority: cpu=pwr11 (500) > power9-vector (311) > power8-vector (212) > vsx (111) > altivec (50) > default (0)
int __attribute__((target_clones("altivec", "power8-vector", "cpu=pwr11", "vsx", "power9-vector", "default")))
foo_multi_features(void) { return 0; }
// CHECK: define internal {{.*}}i32 @foo_multi_features.altivec()
// CHECK: define internal {{.*}}i32 @foo_multi_features.power8_vector()
// CHECK: define internal {{.*}}i32 @foo_multi_features.cpu_pwr11() #[[#ATTR_P11]]
// CHECK: define internal {{.*}}i32 @foo_multi_features.vsx()
// CHECK: define internal {{.*}}i32 @foo_multi_features.power9_vector()
// CHECK: define internal {{.*}}i32 @foo_multi_features.default()
// CHECK: define internal ptr @foo_multi_features.resolver()
//   Resolver checks in priority order (highest first):
//   if (__builtin_cpu_supports("arch_3_1")) return &foo_multi_features.cpu_pwr11;
// CHECK: load i32, ptr getelementptr inbounds nuw (i8, ptr @_system_configuration, {{i32|i64}} 4)
// CHECK-NEXT: icmp uge i32 {{.*}}, 262144
// CHECK: ret ptr @foo_multi_features.cpu_pwr11
//   if (__builtin_cpu_supports("arch_3_00")) return &foo_multi_features.power9_vector;
// CHECK: load i32, ptr getelementptr inbounds nuw (i8, ptr @_system_configuration, {{i32|i64}} 4)
// CHECK-NEXT: icmp uge i32 {{.*}}, 131072
// CHECK: ret ptr @foo_multi_features.power9_vector
//   if (__builtin_cpu_supports("arch_2_07")) return &foo_multi_features.power8_vector;
// CHECK: load i32, ptr getelementptr inbounds nuw (i8, ptr @_system_configuration, {{i32|i64}} 4)
// CHECK-NEXT: icmp uge i32 {{.*}}, 65536
// CHECK: ret ptr @foo_multi_features.power8_vector
//   if (__builtin_cpu_supports("vsx")) return &foo_multi_features.vsx;
// CHECK: %[[#VSX2:]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @_system_configuration, {{i32|i64}} 204)
// CHECK-NEXT: icmp ugt i32 %[[#VSX2]], 1
// CHECK: ret ptr @foo_multi_features.vsx
//   if (__builtin_cpu_supports("altivec")) return &foo_multi_features.altivec;
// CHECK: %[[#ALTIVEC2:]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @_system_configuration, {{i32|i64}} 204)
// CHECK-NEXT: icmp ugt i32 %[[#ALTIVEC2]], 0
// CHECK: ret ptr @foo_multi_features.altivec
// CHECK: ret ptr @foo_multi_features.default

// Test negated feature (no-altivec) - should negate the condition
int __attribute__((target_clones("no-altivec", "default")))
foo_no_altivec(void) { return 0; }
// CHECK: define internal {{.*}}i32 @foo_no_altivec.no_altivec()
// CHECK: define internal {{.*}}i32 @foo_no_altivec.default()
// CHECK: define internal ptr @foo_no_altivec.resolver()
//   if (!__builtin_cpu_supports("altivec")) return &foo_no_altivec.no_altivec;
// CHECK: load i32, ptr getelementptr inbounds nuw (i8, ptr @_system_configuration, {{i32|i64}} 204)
// CHECK-NEXT: icmp ugt i32
// CHECK-NEXT: %neg = xor i1 {{.*}}, true
// CHECK-NEXT: br i1 %neg, label %if.version, label %if.else
// CHECK: if.version:
// CHECK-NEXT: ret ptr @foo_no_altivec.no_altivec
// CHECK: if.else:
// CHECK-NEXT: ret ptr @foo_no_altivec.default

// CHECK: declare i64 @getsystemcfg(i32)

// CHECK: attributes #[[#ATTR_P7]] = {{.*}} "target-cpu"="pwr7"
// CHECK: attributes #[[#ATTR_P10]] = {{.*}} "target-cpu"="pwr10"
// CHECK: attributes #[[#ATTR_P11]] = {{.*}} "target-cpu"="pwr11"
// CHECK: attributes #[[#ATTR_P9]] = {{.*}} "target-cpu"="pwr9"
// CHECK: attributes #[[#ATTR_P8]] = {{.*}} "target-cpu"="pwr8"

