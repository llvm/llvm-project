	
// Test sanitizer link flags on Darwin.

// RUN: %clang -### --target=x86_64-darwin \
// RUN:   -stdlib=platform -fmemory-profile %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MEMPROF %s

// CHECK-MEMPROF: "{{.*}}ld{{(.exe)?}}"
// CHECK-MEMPROF-SAME: libclang_rt.memprof_osx_dynamic.dylib"
// CHECK-MEMPROF-SAME: "-rpath" "@executable_path"
// CHECK-MEMPROF-SAME: "-rpath" "{{[^"]*}}lib{{[^"]*}}darwin"
