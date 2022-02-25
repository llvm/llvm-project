// RUN: %clangxx %s -### -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/gcc_version_parsing_rt_libs --target=x86_64-redhat-linux 2>&1 | FileCheck %s -check-prefix=STDCPLUS
// RUN: %clangxx %s -### -stdlib=libc++ --gcc-toolchain=%S/Inputs/gcc_version_parsing_rt_libs --target=x86_64-redhat-linux 2>&1 | FileCheck %s -check-prefix=LIBCPLUS

int main() {}

// STDCPLUS: "-internal-isystem" "{{[^ ]*}}gcc_version_parsing_rt_libs/lib/gcc/x86_64-redhat-linux/10.2.0/../../../gcc/x86_64-redhat-linux/10.2.0/include/c++/"
// STDCPLUS: "-L{{.*}}gcc_version_parsing_rt_libs/lib/gcc/x86_64-redhat-linux/10.2.0"
// STDCPLUS: "-L{{.*}}gcc_version_parsing_rt_libs/lib/gcc/x86_64-redhat-linux/10.2.0/../lib64"

// LIBCPLUS-NOT: "-internal-isystem" "{{[^ ]*}}gcc_version_parsing_rt_libs/lib/gcc/x86_64-redhat-linux/10.2.0/../../../gcc/x86_64-redhat-linux/10.2.0/include/c++/"
// LIBCPLUS: "-L{{.*}}gcc_version_parsing_rt_libs/lib/gcc/x86_64-redhat-linux/10.2.0"
// LIBCPLUS: "-L{{.*}}gcc_version_parsing_rt_libs/lib/gcc/x86_64-redhat-linux/10.2.0/../lib64"
