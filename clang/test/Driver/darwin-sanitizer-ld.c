// Test sanitizer link flags on Darwin.

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -stdlib=platform -fsanitize=address %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN %s

// CHECK-ASAN: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-NOT: "-lstdc++"
// CHECK-ASAN-NOT: "-lc++"
// CHECK-ASAN: libclang_rt.asan_osx_dynamic.dylib"
// CHECK-ASAN: "-rpath" "@executable_path"
// CHECK-ASAN: "-rpath" "{{.*}}lib{{.*}}darwin"

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -fPIC -shared -fsanitize=address %s -o %t.so 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DYN-ASAN %s

// CHECK-DYN-ASAN: "{{.*}}ld{{(.exe)?}}"
// CHECK-DYN-ASAN: "-dylib"
// CHECK-DYN-ASAN: libclang_rt.asan_osx_dynamic.dylib"
// CHECK-DYN-ASAN: "-rpath" "@executable_path"
// CHECK-DYN-ASAN: "-rpath" "{{.*}}lib{{.*}}darwin"

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -stdlib=platform -fsanitize=undefined %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN %s

// CHECK-UBSAN: "{{.*}}ld{{(.exe)?}}"
// CHECK-UBSAN-NOT: "-lstdc++"
// CHECK-UBSAN-NOT: "-lc++"
// CHECK-UBSAN: libclang_rt.ubsan_osx_dynamic.dylib"
// CHECK-UBSAN: "-rpath" "@executable_path"
// CHECK-UBSAN: "-rpath" "{{.*}}lib{{.*}}darwin"

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -fsanitize=bounds -fsanitize-undefined-trap-on-error \
// RUN:   %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-BOUNDS %s

// CHECK-BOUNDS: "{{.*}}ld{{(.exe)?}}"
// CHECK-BOUNDS-NOT: libclang_rt.ubsan_osx.a"

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -fPIC -shared -fsanitize=undefined %s -o %t.so 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DYN-UBSAN %s

// CHECK-DYN-UBSAN: "{{.*}}ld{{(.exe)?}}"
// CHECK-DYN-UBSAN: "-dylib"
// CHECK-DYN-UBSAN: libclang_rt.ubsan_osx_dynamic.dylib"
// CHECK-DYN-UBSAN: "-rpath" "@executable_path"
// CHECK-DYN-UBSAN: "-rpath" "{{.*}}lib{{.*}}darwin"

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -fsanitize=bounds -fsanitize-undefined-trap-on-error \
// RUN:   %s -o %t.so -fPIC -shared 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DYN-BOUNDS %s

// CHECK-DYN-BOUNDS: "{{.*}}ld{{(.exe)?}}"
// CHECK-DYN-BOUNDS-NOT: ubsan_osx

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -stdlib=platform -fsanitize=address -mios-simulator-version-min=7.0 \
// RUN:   %s -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-ASAN-IOSSIM %s

// CHECK-ASAN-IOSSIM: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-IOSSIM-NOT: "-lstdc++"
// CHECK-ASAN-IOSSIM-NOT: "-lc++"
// CHECK-ASAN-IOSSIM: libclang_rt.asan_iossim_dynamic.dylib"
// CHECK-ASAN-IOSSIM: "-rpath" "@executable_path"
// CHECK-ASAN-IOSSIM: "-rpath" "{{.*}}lib{{.*}}darwin"

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -stdlib=platform -fsanitize=address \
// RUN:   -mtvos-simulator-version-min=8.3.0 %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-TVOSSIM %s

// CHECK-ASAN-TVOSSIM: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-TVOSSIM-NOT: "-lstdc++"
// CHECK-ASAN-TVOSSIM-NOT: "-lc++"
// CHECK-ASAN-TVOSSIM: libclang_rt.asan_tvossim_dynamic.dylib"
// CHECK-ASAN-TVOSSIM: "-rpath" "@executable_path"
// CHECK-ASAN-TVOSSIM: "-rpath" "{{.*}}lib{{.*}}darwin"

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -stdlib=platform -fsanitize=address \
// RUN:   -mwatchos-simulator-version-min=2.0.0 %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN-WATCHOSSIM %s

// CHECK-ASAN-WATCHOSSIM: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-WATCHOSSIM-NOT: "-lstdc++"
// CHECK-ASAN-WATCHOSSIM-NOT: "-lc++"
// CHECK-ASAN-WATCHOSSIM: libclang_rt.asan_watchossim_dynamic.dylib"
// CHECK-ASAN-WATCHOSSIM: "-rpath" "@executable_path"
// CHECK-ASAN-WATCHOSSIM: "-rpath" "{{.*}}lib{{.*}}darwin"

// RUN: %clang -no-canonical-prefixes -### -target x86_64-apple-ios13.1-macabi \
// RUN:   -stdlib=platform -fsanitize=address \
// RUN:   -resource-dir %S/Inputs/resource_dir \
// RUN:   %s -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-ASAN-MACCATALYST %s

// CHECK-ASAN-MACCATALYST: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-MACCATALYST-NOT: "-lstdc++"
// CHECK-ASAN-MACCATALYST-NOT: "-lc++"
// CHECK-ASAN-MACCATALYST: libclang_rt.asan_osx_dynamic.dylib"
// CHECK-ASAN-MACCATALYST: "-rpath" "@executable_path"
// CHECK-ASAN-MACCATALYST: "-rpath" "{{.*}}lib{{.*}}darwin"

// RUN: %clang -no-canonical-prefixes -### -target armv7-apple-ios  \
// RUN:   -stdlib=platform -fsanitize=address -miphoneos-version-min=7 \
// RUN:   %s -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-ASAN-IOS %s

// CHECK-ASAN-IOS: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-IOS-NOT: "-lstdc++"
// CHECK-ASAN-IOS-NOT: "-lc++"
// CHECK-ASAN-IOS: libclang_rt.asan_ios_dynamic.dylib"
// CHECK-ASAN-IOS: "-rpath" "@executable_path"
// CHECK-ASAN-IOS: "-rpath" "{{.*}}lib{{.*}}darwin"

// RUN: %clang -no-canonical-prefixes -### -target arm64-apple-tvos \
// RUN:   -stdlib=platform -fsanitize=address -mtvos-version-min=8.3 \
// RUN:   %s -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-ASAN-TVOS %s

// CHECK-ASAN-TVOS: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-TVOS-NOT: "-lstdc++"
// CHECK-ASAN-TVOS-NOT: "-lc++"
// CHECK-ASAN-TVOS: libclang_rt.asan_tvos_dynamic.dylib"
// CHECK-ASAN-TVOS: "-rpath" "@executable_path"
// CHECK-ASAN-TVOS: "-rpath" "{{.*}}lib{{.*}}darwin"

// RUN: %clang -no-canonical-prefixes -### -target armv7k-apple-watchos \
// RUN:   -stdlib=platform -fsanitize=address -mwatchos-version-min=2.0 \
// RUN:   %s -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-ASAN-WATCHOS %s

// CHECK-ASAN-WATCHOS: "{{.*}}ld{{(.exe)?}}"
// CHECK-ASAN-WATCHOS-NOT: "-lstdc++"
// CHECK-ASAN-WATCHOS-NOT: "-lc++"
// CHECK-ASAN-WATCHOS: libclang_rt.asan_watchos_dynamic.dylib"
// CHECK-ASAN-WATCHOS: "-rpath" "@executable_path"
// CHECK-ASAN-WATCHOS: "-rpath" "{{.*}}lib{{.*}}darwin"
