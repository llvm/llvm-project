// SYCL device code defaults to defined integer division and remainder, except
// on SPIR and SPIR-V targets; host code retains undefined behavior.

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=OFF --implicit-check-not=.poison
// RUN: %clang_cc1 -fsycl-is-device -triple nvptx64-nvidia-cuda -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fdefined-int-divrem -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-unknown-linux-gnu -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=OFF --implicit-check-not=.poison

int f_sdiv(int a, int b) { return a / b; }
// ON-LABEL: define {{.*}} @{{.*}}f_sdiv
// ON: = call i32 @llvm.sdiv.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_sdiv
// OFF: = sdiv i32

int f_srem(int a, int b) { return a % b; }
// ON-LABEL: define {{.*}} @{{.*}}f_srem
// ON: = call i32 @llvm.srem.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_srem
// OFF: = srem i32

unsigned f_udiv(unsigned a, unsigned b) { return a / b; }
// ON-LABEL: define {{.*}} @{{.*}}f_udiv
// ON: = call i32 @llvm.udiv.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_udiv
// OFF: = udiv i32

unsigned f_urem(unsigned a, unsigned b) { return a % b; }
// ON-LABEL: define {{.*}} @{{.*}}f_urem
// ON: = call i32 @llvm.urem.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_urem
// OFF: = urem i32

// A safe constant divisor keeps the plain instruction in both modes.
int f_divc(int a) { return a / 4; }
// ON-LABEL: define {{.*}} @{{.*}}f_divc
// ON: = sdiv i32 {{.*}}, 4
// OFF-LABEL: define {{.*}} @{{.*}}f_divc
// OFF: = sdiv i32 {{.*}}, 4

// Device compilation emits functions reachable from a SYCL kernel.
template <typename Name, typename... Ts> void sycl_kernel_launch(Ts...) {}

template <typename Name, typename Func>
[[clang::sycl_kernel_entry_point(Name)]] void kernel_single_task(const Func &f) {
  f();
}

void launch(int *out, int a, int b) {
  kernel_single_task<class DivRem>([=] {
    out[0] = f_sdiv(a, b);
    out[1] = f_srem(a, b);
    out[2] = f_udiv(a, b);
    out[3] = f_urem(a, b);
    out[4] = f_divc(a);
  });
}
