// RUN: %clang -target arm-linux-gnueabi -rtlib=compiler-rt -### %s 2>&1 | FileCheck %s -check-prefix ARM-GNUEABI
// ARM-GNUEABI: "{{.*[/\\]}}libclang_rt.builtins-arm.a"

// RUN: %clang -target arm-linux-gnueabi -rtlib=compiler-rt -mfloat-abi=hard -### %s 2>&1 | FileCheck %s -check-prefix ARM-GNUEABI-ABI
// ARM-GNUEABI-ABI: "{{.*[/\\]}}libclang_rt.builtins-armhf.a"

// RUN: %clang -target arm-linux-gnueabihf -rtlib=compiler-rt -### %s 2>&1 | FileCheck %s -check-prefix ARM-GNUEABIHF
// ARM-GNUEABIHF: "{{.*[/\\]}}libclang_rt.builtins-armhf.a"

// RUN: %clang -target arm-linux-gnueabihf -rtlib=compiler-rt -mfloat-abi=soft -### %s 2>&1 | FileCheck %s -check-prefix ARM-GNUEABIHF-ABI
// ARM-GNUEABIHF-ABI: "{{.*[/\\]}}libclang_rt.builtins-arm.a"

// RUN: %clang -target arm-windows-itanium -rtlib=compiler-rt -### %s 2>&1 | FileCheck %s -check-prefix ARM-WINDOWS
// ARM-WINDOWS: "{{.*[/\\]}}libclang_rt.builtins-arm.lib"

// RUN: %clang -target arm-linux-androideabi -rtlib=compiler-rt -### %s 2>&1 | FileCheck %s -check-prefix ARM-ANDROID
// ARM-ANDROID: "{{.*[/\\]}}libclang_rt.builtins-arm-android.a"

// RUN: %clang -target arm-linux-androideabi -rtlib=compiler-rt -mfloat-abi=hard -### %s 2>&1 | FileCheck %s -check-prefix ARM-ANDROIDHF
// ARM-ANDROIDHF: "{{.*[/\\]}}libclang_rt.builtins-armhf-android.a"

