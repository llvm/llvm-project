// RUN: %clangxx %s -### -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/gcc_version_parsing_rt_libs_multilib --target=x86_64-redhat-linux 2>&1 | FileCheck %s -check-prefix=X64-STDCPLUS
// RUN: %clangxx %s -### -stdlib=libc++ --gcc-toolchain=%S/Inputs/gcc_version_parsing_rt_libs_multilib --target=x86_64-redhat-linux 2>&1 | FileCheck %s -check-prefix=X64-LIBCPLUS
// RUN: %clangxx %s -m32 -### -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/gcc_version_parsing_rt_libs_multilib --target=x86_64-redhat-linux 2>&1 | FileCheck %s -check-prefix=X32-STDCPLUS
// RUN: %clangxx %s -m32 -### -stdlib=libc++ --gcc-toolchain=%S/Inputs/gcc_version_parsing_rt_libs_multilib --target=x86_64-redhat-linux 2>&1 | FileCheck %s -check-prefix=X32-LIBCPLUS

int main() {}

// X64-STDCPLUS: "-internal-isystem" "{{[^ ]*}}gcc_version_parsing_rt_libs_multilib/lib/gcc/x86_64-redhat-linux/10.2.0/../../../gcc/x86_64-redhat-linux/10.2.0/include/c++/"
// X64-STDCPLUS: "-L{{[^ ]*}}gcc_version_parsing_rt_libs_multilib/lib/gcc/x86_64-redhat-linux/10.2.0"
// X64-STDCPLUS: "-L{{[^ ]*}}gcc_version_parsing_rt_libs_multilib/lib/gcc/x86_64-redhat-linux/10.2.0/../lib64"

// X64-LIBCPLUS-NOT: "-internal-isystem" "{{[^ ]*}}gcc_version_parsing_rt_libs_multilib/lib/gcc/x86_64-redhat-linux/10.2.0/../../../gcc/x86_64-redhat-linux/10.2.0/include/c++/"
// X64-LIBCPLUS: "-L{{[^ ]*}}gcc_version_parsing_rt_libs_multilib/lib/gcc/x86_64-redhat-linux/10.2.0"
// X64-LIBCPLUS: "-L{{[^ ]*}}gcc_version_parsing_rt_libs_multilib/lib/gcc/x86_64-redhat-linux/10.2.0/../lib64"

// X32-STDCPLUS: "-internal-isystem" "{{[^ ]*}}gcc_version_parsing_rt_libs_multilib/lib/gcc/x86_64-redhat-linux/10.2.0/../../../gcc/x86_64-redhat-linux/10.2.0/include/c++/"
// X32-STDCPLUS: "-L{{[^ ]*}}gcc_version_parsing_rt_libs_multilib/lib/gcc/x86_64-redhat-linux/10.2.0/32"
// X32-STDCPLUS: "-L{{[^ ]*}}gcc_version_parsing_rt_libs_multilib/lib/gcc/x86_64-redhat-linux/10.2.0/../lib32"

// X32-LIBCPLUS-NOT: "-internal-isystem" "{{[^ ]*}}gcc_version_parsing_rt_libs_multilib/lib/gcc/x86_64-redhat-linux/10.2.0/../../../gcc/x86_64-redhat-linux/10.2.0/include/c++/"
// X32-LIBCPLUS: "-L{{[^ ]*}}gcc_version_parsing_rt_libs_multilib/lib/gcc/x86_64-redhat-linux/10.2.0/32"
// X32-LIBCPLUS: "-L{{[^ ]*}}gcc_version_parsing_rt_libs_multilib/lib/gcc/x86_64-redhat-linux/10.2.0/../lib32"
