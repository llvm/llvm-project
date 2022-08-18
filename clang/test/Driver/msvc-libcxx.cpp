// RUN: %clangxx -### %s 2>&1 -stdlib=libc++ -fuse-ld=lld \
// RUN:   --target=x86_64-pc-windows-msvc \
// RUN:   -ccc-install-dir %S/Inputs/msvc_libcxx_tree/usr/bin \
// RUN:   | FileCheck %s -check-prefix MSVC-LIBCXX
// MSVC-LIBCXX: "-internal-isystem" "{{.*[/\\]}}include{{/|\\\\}}x86_64-pc-windows-msvc{{/|\\\\}}c++{{/|\\\\}}v1"
// MSVC-LIBCXX: "-internal-isystem" "{{.*[/\\]}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// MSVC-LIBCXX: "-libpath:{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-pc-windows-msvc"
