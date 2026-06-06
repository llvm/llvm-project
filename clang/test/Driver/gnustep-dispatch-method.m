// DEFINE: %{triple} =
// DEFINE: %{ver} = 1.6
// DEFINE: %{prefix} = CHECK-MSGSEND
// DEFINE: %{check} = %clang --target=%{triple} -fobjc-runtime=gnustep-%{ver} -### -c %s 2>&1 | FileCheck -check-prefix=%{prefix} %s

// REDEFINE: %{ver} = 1.6
// REDEFINE: %{triple} = i386-unknown-freebsd
// RUN: %{check}
// REDEFINE: %{triple} = x86_64-unknown-freebsd
// RUN: %{check}
// REDEFINE: %{triple} = arm-unknown-freebsd
// RUN: %{check}
// REDEFINE: %{prefix} = CHECK-MSGLOOKUP
// REDEFINE: %{triple} = aarch64-unknown-freebsd
// RUN: %{check}
// REDEFINE: %{triple} = mips64-unknown-freebsd
// RUN: %{check}
// REDEFINE: %{triple} = riscv64-unknown-freebsd
// RUN: %{check}

// REDEFINE: %{ver} = 1.9
// REDEFINE: %{prefix} = CHECK-MSGSEND
// REDEFINE: %{triple} = aarch64-unknown-freebsd
// RUN: %{check}
// REDEFINE: %{triple} = mips64-unknown-freebsd
// RUN: %{check}
// REDEFINE: %{prefix} = CHECK-MSGLOOKUP
// REDEFINE: %{triple} = riscv64-unknown-freebsd
// RUN: %{check}

// REDEFINE: %{ver} = 2.2
// REDEFINE: %{prefix} = CHECK-MSGSEND
// REDEFINE: %{triple} = riscv64-unknown-freebsd
// RUN: %{check}


// CHECK-MSGSEND: "-cc1"{{.*}} "-fobjc-dispatch-method=non-legacy"
// CHECK-MSGLOOKUP-NOT: "-cc1"{{.*}} "-fobjc-dispatch-method=non-legacy"
