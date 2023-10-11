// Test sanitizer link flags on Darwin.

// RUN: %clang -### --target=x86_64-darwin \
// RUN:   -stdlib=platform -fmemory-profile %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MEMPROF %s

// CHECK-MEMPROF: "{{.*}}ld{{(.exe)?}}"
// CHECK-MEMPROF-NOT: "-lstdc++"
// CHECK-MEMPROF-NOT: "-lc++"
// CHECK-MEMPROF: libclang_rt.memprof_osx_dynamic.dylib"
// CHECK-MEMPROF: "-rpath" "@executable_path"
// CHECK-MEMPROF: "-rpath" "{{.*}}lib{{.*}}darwin"

