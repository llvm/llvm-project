// RUN: %clang --target=aarch64-pc-windows-msvc -O3 -flto -fuse-ld=lld -### %s 2>&1 | FileCheck %s
// RUN: %clang --target=aarch64-pc-windows-msvc -O3 -flto=thin -fuse-ld=lld -### %s 2>&1 | FileCheck %s

// RUN: %clang --target=aarch64-pc-windows-msvc -O3 -flto -fuse-ld=lld-link -### %s 2>&1 | FileCheck %s
// RUN: %clang --target=aarch64-pc-windows-msvc -O3 -flto=thin -fuse-ld=lld-link -### %s 2>&1 | FileCheck %s

// CHECK: "{{.*}}lld-link{{(.exe)?}}" "-out:a.exe" "-defaultlib:libcmt" "-defaultlib:oldnames"

int main() { return 0; }
