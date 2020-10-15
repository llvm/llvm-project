// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu i686 -tune-cpu i686 -emit-llvm %s -o - | FileCheck %s

int baz(int a) { return 4; }

int __attribute__((target("avx,sse4.2,arch=ivybridge"))) foo(int a) { return 4; }

int __attribute__((target("fpmath=387"))) koala(int a) { return 4; }

int __attribute__((target("no-sse2"))) echidna(int a) { return 4; }

int __attribute__((target("sse4"))) panda(int a) { return 4; }
int __attribute__((target("no-sse4"))) narwhal(int a) { return 4; }

int bar(int a) { return baz(a) + foo(a); }

int __attribute__((target("avx,      sse4.2,      arch=   ivybridge"))) qux(int a) { return 4; }
int __attribute__((target("no-aes, arch=ivybridge"))) qax(int a) { return 4; }

int __attribute__((target("no-mmx"))) qq(int a) { return 40; }

int __attribute__((target("arch=lakemont,mmx"))) lake(int a) { return 4; }

int use_before_def(void);
int useage(void){
  return use_before_def();
}

// Adding the attribute to a definition does update it in IR.
int __attribute__((target("arch=lakemont,mmx"))) use_before_def(void) {
  return 5;
}

int __attribute__((target("tune=sandybridge"))) walrus(int a) { return 4; }

void __attribute__((target("arch=x86-64-v2"))) x86_64_v2() {}
void __attribute__((target("arch=x86-64-v3"))) x86_64_v3() {}
void __attribute__((target("arch=x86-64-v4"))) x86_64_v4() {}

// Check that we emit the additional subtarget and cpu features for foo and not for baz or bar.
// CHECK: baz{{.*}} #0
// CHECK: foo{{.*}} #1
// We're currently ignoring the fpmath attribute so koala should be identical to baz and bar.
// CHECK: koala{{.*}} #0
// CHECK: echidna{{.*}} #2
// CHECK: panda{{.*}} #3
// CHECK: narwhal{{.*}} #4
// CHECK: bar{{.*}} #0
// CHECK: qux{{.*}} #1
// CHECK: qax{{.*}} #5
// CHECK: qq{{.*}} #6
// CHECK: lake{{.*}} #7
// CHECK: use_before_def{{.*}} #7
// CHECK: walrus{{.*}} #8
// CHECK: #0 = {{.*}}"target-cpu"="i686" "target-features"="+cx8,+x87" "tune-cpu"="i686"
// CHECK: #1 = {{.*}}"target-cpu"="ivybridge" "target-features"="+avx,+cx16,+cx8,+f16c,+fsgsbase,+fxsr,+mmx,+pclmul,+popcnt,+rdrnd,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt"
// CHECK-NOT: tune-cpu
// CHECK: #2 = {{.*}}"target-cpu"="i686" "target-features"="+cx8,+x87,-aes,-avx,-avx2,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-f16c,-fma,-fma4,-gfni,-kl,-pclmul,-sha,-sse2,-sse3,-sse4.1,-sse4.2,-sse4a,-ssse3,-vaes,-vpclmulqdq,-widekl,-xop" "tune-cpu"="i686"
// CHECK: #3 = {{.*}}"target-cpu"="i686" "target-features"="+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87" "tune-cpu"="i686"
// CHECK: #4 = {{.*}}"target-cpu"="i686" "target-features"="+cx8,+x87,-avx,-avx2,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-f16c,-fma,-fma4,-sse4.1,-sse4.2,-vaes,-vpclmulqdq,-xop" "tune-cpu"="i686"
// CHECK: #5 = {{.*}}"target-cpu"="ivybridge" "target-features"="+avx,+cx16,+cx8,+f16c,+fsgsbase,+fxsr,+mmx,+pclmul,+popcnt,+rdrnd,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-aes,-vaes"
// CHECK-NOT: tune-cpu
// CHECK: #6 = {{.*}}"target-cpu"="i686" "target-features"="+cx8,+x87,-3dnow,-3dnowa,-mmx"
// CHECK: #7 = {{.*}}"target-cpu"="lakemont" "target-features"="+cx8,+mmx"
// CHECK-NOT: tune-cpu
// CHECK: #8 = {{.*}}"target-cpu"="i686" "target-features"="+cx8,+x87" "tune-cpu"="sandybridge"

// CHECK: "target-cpu"="x86-64-v2"
// CHECK-SAME: "target-features"="+cx16,+cx8,+fxsr,+mmx,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87"
// CHECK: "target-cpu"="x86-64-v3"
// CHECK-SAME: "target-features"="+avx,+avx2,+bmi,+bmi2,+cx16,+cx8,+f16c,+fma,+fxsr,+lzcnt,+mmx,+movbe,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
// CHECK: "target-cpu"="x86-64-v4"
// CHECK-SAME: "target-features"="+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+cx16,+cx8,+f16c,+fma,+fxsr,+lzcnt,+mmx,+movbe,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
