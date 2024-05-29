// NOTE: See arm-cortex-cpus-2.c for additional tests The tests are split
// across multiple files, to avoid excessive test times for large single
// test files.
// TODO: The files should be split up by categories, e.g. by architecture versions.
//
// ================== Check default CPU on each major architecture
// RUN: %clang -target arm -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-GENERIC %s
// CHECK-GENERIC: "-cc1"{{.*}} "-triple" "armv4t-{{.*}} "-target-cpu" "generic"

// RUN: %clang -target armeb -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-GENERIC %s
// CHECK-BE-GENERIC: "-cc1"{{.*}} "-triple" "armebv4t-{{.*}} "-target-cpu" "generic"

// RUN: %clang -target arm -mthumb -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-GENERIC-THUMB %s
// CHECK-GENERIC-THUMB: "-cc1"{{.*}} "-triple" "thumbv4t-{{.*}} "-target-cpu" "generic"

// RUN: %clang -target armeb -mthumb -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-GENERIC-THUMB %s
// CHECK-BE-GENERIC-THUMB: "-cc1"{{.*}} "-triple" "thumbebv4t-{{.*}} "-target-cpu" "generic"

// RUN: %clang -target armv4t -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V4T %s
// RUN: %clang -target arm -march=armv4t -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V4T %s
// CHECK-V4T: "-cc1"{{.*}} "-triple" "armv4t-{{.*}} "-target-cpu" "arm7tdmi"

// RUN: %clang -target armv4t -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V4T-THUMB %s
// RUN: %clang -target arm -mthumb -march=armv4t -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V4T-THUMB %s
// CHECK-V4T-THUMB: "-cc1"{{.*}} "-triple" "thumbv4t-{{.*}} "-target-cpu" "arm7tdmi"

// RUN: %clang -target armv5 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5 %s
// RUN: %clang -target arm -march=armv5 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5 %s
// RUN: %clang -target armv5t -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5 %s
// RUN: %clang -target arm -march=armv5t -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5 %s
// CHECK-V5: "-cc1"{{.*}} "-triple" "armv5-{{.*}} "-target-cpu" "arm10tdmi"

// RUN: %clang -target armv5 -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5-THUMB %s
// RUN: %clang -target arm -march=armv5 -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5-THUMB %s
// RUN: %clang -target armv5t -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5-THUMB %s
// CHECK-V5-THUMB: "-cc1"{{.*}} "-triple" "thumbv5-{{.*}} "-target-cpu" "arm10tdmi"

// RUN: %clang -target armv5e -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5E %s
// RUN: %clang -target arm -march=armv5e -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5E %s
// CHECK-V5E: "-cc1"{{.*}} "-triple" "armv5e-{{.*}} "-target-cpu" "arm1022e"

// RUN: %clang -target armv5e -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5E-THUMB %s
// RUN: %clang -target arm -march=armv5e -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5E-THUMB %s
// CHECK-V5E-THUMB: "-cc1"{{.*}} "-triple" "thumbv5e-{{.*}} "-target-cpu" "arm1022e"

// FIXME %clang -target armv5tej -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5TEJ %s
// RUN: %clang -target arm -march=armv5tej -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5TEJ %s
// CHECK-V5TEJ: "-cc1"{{.*}} "-triple" "armv5e-{{.*}} "-target-cpu" "arm926ej-s"

// FIXME %clang -target armv5tej -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5TEJ-THUMB %s
// RUN: %clang -target arm -march=armv5tej -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V5TEJ-THUMB %s
// CHECK-V5TEJ-THUMB: "-cc1"{{.*}} "-triple" "thumbv5e-{{.*}} "-target-cpu" "arm926ej-s"

// RUN: %clang -target armv6 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6 %s
// RUN: %clang -target arm -march=armv6 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6 %s
// CHECK-V6: "-cc1"{{.*}} "-triple" "armv6-{{.*}} "-target-cpu" "arm1136jf-s"

// RUN: %clang -target armv6 -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6-THUMB %s
// RUN: %clang -target arm -march=armv6 -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6-THUMB %s
// CHECK-V6-THUMB: "-cc1"{{.*}} "-triple" "thumbv6-{{.*}} "-target-cpu" "arm1136jf-s"

// FIXME %clang -target armv6j -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6J %s
// RUN: %clang -target arm -march=armv6j -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6J %s
// CHECK-V6J: "-cc1"{{.*}} "-triple" "armv6-{{.*}} "-target-cpu" "arm1136jf-s"

// FIXME %clang -target armv6j -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6J-THUMB %s
// RUN: %clang -target arm -march=armv6j -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6J-THUMB %s
// CHECK-V6J-THUMB: "-cc1"{{.*}} "-triple" "thumbv6-{{.*}} "-target-cpu" "arm1136jf-s"

// FIXME %clang -target armv6z -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6Z %s
// FIXME %clang -target arm -march=armv6z -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6Z %s
// CHECK-V6Z: "-cc1"{{.*}} "-triple" "armv6-{{.*}} "-target-cpu" "arm1176jzf-s"

// FIXME %clang -target armv6z -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6Z-THUMB %s
// FIXME %clang -target arm -march=armv6z -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6Z-THUMB %s
// CHECK-V6Z-THUMB: "-cc1"{{.*}} "-triple" "thumbv6-{{.*}} "-target-cpu" "arm1176jzf-s"

// RUN: %clang -target armv6k -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6K %s
// RUN: %clang -target arm -march=armv6k -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6K %s
// CHECK-V6K: "-cc1"{{.*}} "-triple" "armv6k-{{.*}} "-target-cpu" "mpcore"

// RUN: %clang -target armv6k -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6K-THUMB %s
// RUN: %clang -target arm -march=armv6k -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6K-THUMB %s
// CHECK-V6K-THUMB: "-cc1"{{.*}} "-triple" "thumbv6k-{{.*}} "-target-cpu" "mpcore"

// RUN: %clang -target armv6t2 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6T2 %s
// RUN: %clang -target arm -march=armv6t2 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6T2 %s
// CHECK-V6T2: "-cc1"{{.*}} "-triple" "armv6t2-{{.*}} "-target-cpu" "arm1156t2-s"

// RUN: %clang -target armv6t2 -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6T2-THUMB %s
// RUN: %clang -target arm -march=armv6t2 -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6T2-THUMB %s
// CHECK-V6T2-THUMB: "-cc1"{{.*}} "-triple" "thumbv6t2-{{.*}} "-target-cpu" "arm1156t2-s"

// RUN: %clang -target armv6m -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6M %s
// RUN: %clang -target arm -march=armv6m -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6M %s
// RUN: %clang -target armv6sm -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6M %s
// RUN: %clang -target arm -march=armv6sm -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6M %s
// CHECK-V6M: "-cc1"{{.*}} "-triple" "thumbv6m-{{.*}} "-target-cpu" "cortex-m0"

// RUN: %clang -target armv6m -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6M-BIG %s
// RUN: %clang -target arm -march=armv6m -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6M-BIG %s
// CHECK-V6M-BIG: "-cc1"{{.*}} "-triple" "thumbebv6m-{{.*}} "-target-cpu" "cortex-m0"

// RUN: %clang -target armv7m -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7M %s
// RUN: %clang -target arm -march=armv7-m -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7M %s
// CHECK-V7M: "-cc1"{{.*}} "-triple" "thumbv7m-{{.*}} "-target-cpu" "cortex-m3"

// RUN: %clang -target armv7em -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7EM %s
// RUN: %clang -target arm -march=armv7e-m -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7EM %s
// CHECK-V7EM: "-cc1"{{.*}} "-triple" "thumbv7em-{{.*}} "-target-cpu" "cortex-m4"

// RUN: %clang -target armv7em -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7EM-BIG %s
// RUN: %clang -target arm -march=armv7e-m -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7EM-BIG %s
// CHECK-V7EM-BIG: "-cc1"{{.*}} "-triple" "thumbebv7em-{{.*}} "-target-cpu" "cortex-m4"

// RUN: %clang -target armv6m-apple-darwin -arch armv6m -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V6M-DARWIN %s
// CHECK-V6M-DARWIN: "-cc1"{{.*}} "-triple" "thumbv6m-{{.*}} "-target-cpu" "cortex-m0"

// RUN: %clang -target armv7m-apple-darwin -arch armv7m -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7M-DARWIN %s
// CHECK-V7M-DARWIN: "-cc1"{{.*}} "-triple" "thumbv7m-{{.*}} "-target-cpu" "cortex-m3"

// RUN: %clang -target armv7em-apple-darwin -arch armv7em -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7EM-DARWIN %s
// CHECK-V7EM-DARWIN: "-cc1"{{.*}} "-triple" "thumbv7em-{{.*}} "-target-cpu" "cortex-m4"

// RUN: %clang -target armv7a-linux-gnueabi -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7A %s
// RUN: %clang -target arm-linux-gnueabi -march=armv7-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7A %s
// CHECK-V7A: "-cc1"{{.*}} "-triple" "armv7-{{.*}} "-target-cpu" "generic"

// RUN: %clang -target armv7a-linux-gnueabi -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7A-THUMB %s
// RUN: %clang -target arm-linux-gnueabi -march=armv7-a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7A-THUMB %s
// CHECK-V7A-THUMB: "-cc1"{{.*}} "-triple" "thumbv7-{{.*}} "-target-cpu" "generic"

// RUN: %clang -target armv7r-linux-gnueabi -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7R %s
// RUN: %clang -target arm-linux-gnueabi -march=armv7-r -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7R %s
// CHECK-V7R: "-cc1"{{.*}} "-triple" "armv7r-{{.*}} "-target-cpu" "cortex-r4"

// RUN: %clang -target armv7r-linux-gnueabi -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7R-THUMB %s
// RUN: %clang -target arm-linux-gnueabi -march=armv7-r -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7R-THUMB %s
// CHECK-V7R-THUMB: "-cc1"{{.*}} "-triple" "thumbv7r-{{.*}} "-target-cpu" "cortex-r4"

// RUN: %clang -target armv8 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A %s
// RUN: %clang -target arm -march=armv8 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A %s
// RUN: %clang -target armv8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A %s
// RUN: %clang -target arm -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A %s
// RUN: %clang -target arm -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A %s
// RUN: %clang -target armv8 -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A %s
// RUN: %clang -target arm -march=armv8 -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A %s
// RUN: %clang -target armv8a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A %s
// RUN: %clang -target arm -march=armv8a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A %s
// RUN: %clang -target arm -march=armv8-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A %s
// CHECK-V8A: "-cc1"{{.*}} "-triple" "armv8a-{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv8r-linux-gnueabi -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8R %s
// RUN: %clang -target arm -march=armv8r -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8R %s
// RUN: %clang -target arm -march=armv8-r -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8R %s
// CHECK-V8R: "-cc1"{{.*}} "-triple" "armv8r-{{.*}} "-target-cpu" "generic"

// RUN: %clang -target armv8r-linux-gnueabi -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8R-BIG %s
// RUN: %clang -target arm -march=armv8r -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8R-BIG %s
// RUN: %clang -target arm -march=armv8-r -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8R-BIG %s
// CHECK-V8R-BIG: "-cc1"{{.*}} "-triple" "armebv8r-{{.*}} "-target-cpu" "generic"

// RUN: %clang -target armv8r-linux-gnueabi -mthumb -### -c %s 2>&1 | \
// RUN:     FileCheck -check-prefix=CHECK-V8R-THUMB %s
// RUN: %clang -target arm -march=armv8r -mthumb -### -c %s 2>&1 | \
// RUN:     FileCheck -check-prefix=CHECK-V8R-THUMB %s
// CHECK-V8R-THUMB: "-cc1"{{.*}} "-triple" "thumbv8r-{{.*}} "-target-cpu" "generic"
// RUN: %clang -target armv8r-linux-gnueabi -mthumb -mbig-endian -### -c %s 2>&1 | \
// RUN:     FileCheck -check-prefix=CHECK-V8R-THUMB-BIG %s
// RUN: %clang -target arm -march=armv8r -mthumb -mbig-endian -### -c %s 2>&1 | \
// RUN:     FileCheck -check-prefix=CHECK-V8R-THUMB-BIG %s
// CHECK-V8R-THUMB-BIG: "-cc1"{{.*}} "-triple" "thumbebv8r-{{.*}} "-target-cpu" "generic"

// RUN: %clang -mcpu=generic -target armv8 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-GENERIC %s
// RUN: %clang -mcpu=generic -target arm -march=armv8 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-GENERIC %s
// RUN: %clang -mcpu=generic -target armv8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-GENERIC %s
// RUN: %clang -mcpu=generic -target arm -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-GENERIC %s
// RUN: %clang -mcpu=generic -target arm -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-GENERIC %s
// RUN: %clang -mcpu=generic -target armv8 -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-GENERIC %s
// RUN: %clang -mcpu=generic -target arm -march=armv8 -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-GENERIC %s
// RUN: %clang -mcpu=generic -target armv8a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-GENERIC %s
// RUN: %clang -mcpu=generic -target arm -march=armv8a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-GENERIC %s
// RUN: %clang -mcpu=generic -target arm -march=armv8-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-GENERIC %s
// CHECK-V8A-GENERIC: "-cc1"{{.*}} "-triple" "armv8a-{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A %s
// RUN: %clang -target armeb -march=armebv8 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A %s
// RUN: %clang -target armebv8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A %s
// RUN: %clang -target armeb -march=armebv8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A %s
// RUN: %clang -target armeb -march=armebv8-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A %s
// RUN: %clang -target armv8 -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A %s
// RUN: %clang -target arm -march=armebv8 -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A %s
// RUN: %clang -target armv8a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A %s
// RUN: %clang -target arm -march=armebv8a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A %s
// RUN: %clang -target arm -march=armebv8-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A %s
// CHECK-BE-V8A: "-cc1"{{.*}} "-triple" "armebv8a-{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv8 -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-THUMB %s
// RUN: %clang -target arm -march=armv8 -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-THUMB %s
// RUN: %clang -target armv8a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-THUMB %s
// RUN: %clang -target arm -march=armv8a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-THUMB %s
// RUN: %clang -target armv8 -mlittle-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-THUMB %s
// RUN: %clang -target arm -march=armv8 -mlittle-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-THUMB %s
// RUN: %clang -target armv8a -mlittle-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-THUMB %s
// RUN: %clang -target arm -march=armv8a -mlittle-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V8A-THUMB %s
// CHECK-V8A-THUMB: "-cc1"{{.*}} "-triple" "thumbv8a-{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8 -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A-THUMB %s
// RUN: %clang -target armeb -march=armebv8 -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A-THUMB %s
// RUN: %clang -target armebv8a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A-THUMB %s
// RUN: %clang -target armeb -march=armebv8a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A-THUMB %s
// RUN: %clang -target armv8 -mbig-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A-THUMB %s
// RUN: %clang -target arm -march=armebv8 -mbig-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A-THUMB %s
// RUN: %clang -target armv8a -mbig-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A-THUMB %s
// RUN: %clang -target arm -march=armebv8a -mbig-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V8A-THUMB %s
// CHECK-BE-V8A-THUMB: "-cc1"{{.*}} "-triple" "thumbebv8a-{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target arm -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A %s
// RUN: %clang -target armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A %s
// RUN: %clang -target arm -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A %s
// RUN: %clang -target armv8.1a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A %s
// RUN: %clang -target arm -march=armv8.1a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A %s
// RUN: %clang -target arm -march=armv8.1-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A %s
// RUN: %clang -mcpu=generic -target arm -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A %s
// RUN: %clang -mcpu=generic -target armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A %s
// RUN: %clang -mcpu=generic -target arm -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A %s
// RUN: %clang -mcpu=generic -target armv8.1a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A %s
// RUN: %clang -mcpu=generic -target arm -march=armv8.1a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A %s
// RUN: %clang -mcpu=generic -target arm -march=armv8.1-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A %s
// CHECK-V81A: "-cc1"{{.*}} "-triple" "armv8.1a-{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V81A %s
// RUN: %clang -target armeb -march=armebv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V81A %s
// RUN: %clang -target armeb -march=armebv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V81A %s
// RUN: %clang -target armv8.1a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V81A %s
// RUN: %clang -target arm -march=armebv8.1a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V81A %s
// RUN: %clang -target arm -march=armebv8.1-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V81A %s
// CHECK-BE-V81A: "-cc1"{{.*}} "-triple" "armebv8.1a-{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv8.1a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A-THUMB %s
// RUN: %clang -target arm -march=armv8.1a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A-THUMB %s
// RUN: %clang -target arm -march=armv8.1-a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A-THUMB %s
// RUN: %clang -target armv8.1a -mlittle-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A-THUMB %s
// RUN: %clang -target arm -march=armv8.1a -mlittle-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A-THUMB %s
// RUN: %clang -target arm -march=armv8.1-a -mlittle-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A-THUMB %s
// CHECK-V81A-THUMB: "-cc1"{{.*}} "-triple" "thumbv8.1a-{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8.1a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V81A-THUMB %s
// RUN: %clang -target armeb -march=armebv8.1a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V81A-THUMB %s
// RUN: %clang -target armeb -march=armebv8.1-a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V81A-THUMB %s
// RUN: %clang -target armv8.1a -mbig-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V81A-THUMB %s
// RUN: %clang -target arm -march=armebv8.1a -mbig-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V81A-THUMB %s
// RUN: %clang -target arm -march=armebv8.1-a -mbig-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V81A-THUMB %s
// CHECK-BE-V81A-THUMB: "-cc1"{{.*}} "-triple" "thumbebv8.1a-{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V82A %s
// RUN: %clang -target arm -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V82A %s
// RUN: %clang -target arm -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V82A %s
// RUN: %clang -target armv8.2a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V82A %s
// RUN: %clang -target arm -march=armv8.2a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V82A %s
// RUN: %clang -target arm -march=armv8.2-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V82A %s
// CHECK-V82A: "-cc1"{{.*}} "-triple" "armv8.2{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V82A %s
// RUN: %clang -target armv8.2a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V82A %s
// RUN: %clang -target armeb -march=armebv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V82A %s
// RUN: %clang -target armeb -march=armebv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V82A %s
// RUN: %clang -target arm -march=armebv8.2a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V82A %s
// RUN: %clang -target arm -march=armebv8.2-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V82A %s
// CHECK-BE-V82A: "-cc1"{{.*}} "-triple" "armebv8.2{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv8.2a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V82A-THUMB %s
// RUN: %clang -target arm -march=armv8.2a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V82A-THUMB %s
// RUN: %clang -target arm -march=armv8.2-a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V82A-THUMB %s
// RUN: %clang -target armv8.2a -mlittle-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V82A-THUMB %s
// RUN: %clang -target arm -march=armv8.2a -mlittle-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V82A-THUMB %s
// RUN: %clang -target arm -march=armv8.2-a -mlittle-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V82A-THUMB %s
// CHECK-V82A-THUMB: "-cc1"{{.*}} "-triple" "thumbv8.2a-{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8.2a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V82A-THUMB %s
// RUN: %clang -target armeb -march=armebv8.2a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V82A-THUMB %s
// RUN: %clang -target armeb -march=armebv8.2-a -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V82A-THUMB %s
// RUN: %clang -target armv8.2a -mbig-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V82A-THUMB %s
// RUN: %clang -target arm -march=armebv8.2a -mbig-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V82A-THUMB %s
// RUN: %clang -target arm -march=armebv8.2-a -mbig-endian -mthumb -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V82A-THUMB %s
// CHECK-BE-V82A-THUMB: "-cc1"{{.*}} "-triple" "thumbebv8.2a-{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V83A %s
// RUN: %clang -target arm -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V83A %s
// RUN: %clang -target arm -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V83A %s
// RUN: %clang -target armv8.3a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V83A %s
// RUN: %clang -target arm -march=armv8.3a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V83A %s
// RUN: %clang -target arm -march=armv8.3-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V83A %s
// CHECK-V83A: "-cc1"{{.*}} "-triple" "armv8.3{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V83A %s
// RUN: %clang -target armv8.3a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V83A %s
// RUN: %clang -target armeb -march=armebv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V83A %s
// RUN: %clang -target armeb -march=armebv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V83A %s
// RUN: %clang -target arm -march=armebv8.3a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V83A %s
// RUN: %clang -target arm -march=armebv8.3-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V83A %s
// CHECK-BE-V83A: "-cc1"{{.*}} "-triple" "armebv8.3{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V84A %s
// RUN: %clang -target arm -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V84A %s
// RUN: %clang -target arm -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V84A %s
// RUN: %clang -target armv8.4a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V84A %s
// RUN: %clang -target arm -march=armv8.4a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V84A %s
// RUN: %clang -target arm -march=armv8.4-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V84A %s
// CHECK-V84A: "-cc1"{{.*}} "-triple" "armv8.4{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V84A %s
// RUN: %clang -target armv8.4a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V84A %s
// RUN: %clang -target armeb -march=armebv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V84A %s
// RUN: %clang -target armeb -march=armebv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V84A %s
// RUN: %clang -target arm -march=armebv8.4a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V84A %s
// RUN: %clang -target arm -march=armebv8.4-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V84A %s
// CHECK-BE-V84A: "-cc1"{{.*}} "-triple" "armebv8.4{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V85A %s
// RUN: %clang -target arm -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V85A %s
// RUN: %clang -target arm -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V85A %s
// RUN: %clang -target armv8.5a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V85A %s
// RUN: %clang -target arm -march=armv8.5a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V85A %s
// RUN: %clang -target arm -march=armv8.5-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V85A %s
// CHECK-V85A: "-cc1"{{.*}} "-triple" "armv8.5{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V85A %s
// RUN: %clang -target armv8.5a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V85A %s
// RUN: %clang -target armeb -march=armebv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V85A %s
// RUN: %clang -target armeb -march=armebv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V85A %s
// RUN: %clang -target arm -march=armebv8.5a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V85A %s
// RUN: %clang -target arm -march=armebv8.5-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V85A %s
// CHECK-BE-V85A: "-cc1"{{.*}} "-triple" "armebv8.5{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V86A %s
// RUN: %clang -target arm -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V86A %s
// RUN: %clang -target arm -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V86A %s
// RUN: %clang -target armv8.6a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V86A %s
// RUN: %clang -target arm -march=armv8.6a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V86A %s
// RUN: %clang -target arm -march=armv8.6-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V86A %s
// CHECK-V86A: "-cc1"{{.*}} "-triple" "armv8.6{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V86A %s
// RUN: %clang -target armv8.6a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V86A %s
// RUN: %clang -target armeb -march=armebv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V86A %s
// RUN: %clang -target armeb -march=armebv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V86A %s
// RUN: %clang -target arm -march=armebv8.6a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V86A %s
// RUN: %clang -target arm -march=armebv8.6-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V86A %s
// CHECK-BE-V86A: "-cc1"{{.*}} "-triple" "armebv8.6{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V87A %s
// RUN: %clang -target arm -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V87A %s
// RUN: %clang -target arm -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V87A %s
// RUN: %clang -target armv8.7a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V87A %s
// RUN: %clang -target arm -march=armv8.7a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V87A %s
// RUN: %clang -target arm -march=armv8.7-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V87A %s
// CHECK-V87A: "-cc1"{{.*}} "-triple" "armv8.7{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V87A %s
// RUN: %clang -target armv8.7a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V87A %s
// RUN: %clang -target armeb -march=armebv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V87A %s
// RUN: %clang -target armeb -march=armebv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V87A %s
// RUN: %clang -target arm -march=armebv8.7a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V87A %s
// RUN: %clang -target arm -march=armebv8.7-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V87A %s
// CHECK-BE-V87A: "-cc1"{{.*}} "-triple" "armebv8.7{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V88A %s
// RUN: %clang -target arm -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V88A %s
// RUN: %clang -target arm -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V88A %s
// RUN: %clang -target armv8.8a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V88A %s
// RUN: %clang -target arm -march=armv8.8a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V88A %s
// RUN: %clang -target arm -march=armv8.8-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V88A %s
// CHECK-V88A: "-cc1"{{.*}} "-triple" "armv8.8{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V88A %s
// RUN: %clang -target armv8.8a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V88A %s
// RUN: %clang -target armeb -march=armebv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V88A %s
// RUN: %clang -target armeb -march=armebv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V88A %s
// RUN: %clang -target arm -march=armebv8.8a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V88A %s
// RUN: %clang -target arm -march=armebv8.8-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V88A %s
// CHECK-BE-V88A: "-cc1"{{.*}} "-triple" "armebv8.8{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv8.9a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V89A %s
// RUN: %clang -target arm -march=armv8.9a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V89A %s
// RUN: %clang -target arm -march=armv8.9-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V89A %s
// RUN: %clang -target arm -march=armv8.9a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V89A %s
// RUN: %clang -target armv8.9a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V89A %s
// RUN: %clang -target arm -march=armv8.9a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V89A %s
// RUN: %clang -target arm -mlittle-endian -march=armv8.9-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V89A %s
// CHECK-V89A: "-cc1"{{.*}} "-triple" "armv8.9{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv8.9a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V89A %s
// RUN: %clang -target armv8.9a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V89A %s
// RUN: %clang -target armeb -march=armebv8.9a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V89A %s
// RUN: %clang -target armeb -march=armebv8.9-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V89A %s
// RUN: %clang -target arm -march=armebv8.9a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V89A %s
// RUN: %clang -target arm -march=armebv8.9-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V89A %s
// CHECK-BE-V89A: "-cc1"{{.*}} "-triple" "armebv8.9{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv9a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V9A %s
// RUN: %clang -target arm -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V9A %s
// RUN: %clang -target arm -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V9A %s
// RUN: %clang -target armv9a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V9A %s
// RUN: %clang -target arm -march=armv9a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V9A %s
// RUN: %clang -target arm -march=armv9-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V9A %s
// CHECK-V9A: "-cc1"{{.*}} "-triple" "armv9{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv9a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V9A %s
// RUN: %clang -target armv9a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V9A %s
// RUN: %clang -target armeb -march=armebv9a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V9A %s
// RUN: %clang -target armeb -march=armebv9-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V9A %s
// RUN: %clang -target arm -march=armebv9a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V9A %s
// RUN: %clang -target arm -march=armebv9-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V9A %s
// CHECK-BE-V9A: "-cc1"{{.*}} "-triple" "armebv9{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V91A %s
// RUN: %clang -target arm -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V91A %s
// RUN: %clang -target arm -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V91A %s
// RUN: %clang -target armv9.1a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V91A %s
// RUN: %clang -target arm -march=armv9.1a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V91A %s
// RUN: %clang -target arm -march=armv9.1-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V91A %s
// CHECK-V91A: "-cc1"{{.*}} "-triple" "armv9.1{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V91A %s
// RUN: %clang -target armv9.1a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V91A %s
// RUN: %clang -target armeb -march=armebv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V91A %s
// RUN: %clang -target armeb -march=armebv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V91A %s
// RUN: %clang -target arm -march=armebv9.1a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V91A %s
// RUN: %clang -target arm -march=armebv9.1-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V91A %s
// CHECK-BE-V91A: "-cc1"{{.*}} "-triple" "armebv9.1{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V92A %s
// RUN: %clang -target arm -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V92A %s
// RUN: %clang -target arm -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V92A %s
// RUN: %clang -target armv9.2a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V92A %s
// RUN: %clang -target arm -march=armv9.2a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V92A %s
// RUN: %clang -target arm -march=armv9.2-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V92A %s
// CHECK-V92A: "-cc1"{{.*}} "-triple" "armv9.2{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V92A %s
// RUN: %clang -target armv9.2a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V92A %s
// RUN: %clang -target armeb -march=armebv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V92A %s
// RUN: %clang -target armeb -march=armebv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V92A %s
// RUN: %clang -target arm -march=armebv9.2a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V92A %s
// RUN: %clang -target arm -march=armebv9.2-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V92A %s
// CHECK-BE-V92A: "-cc1"{{.*}} "-triple" "armebv9.2{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V93A %s
// RUN: %clang -target arm -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V93A %s
// RUN: %clang -target arm -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V93A %s
// RUN: %clang -target armv9.3a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V93A %s
// RUN: %clang -target arm -march=armv9.3a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V93A %s
// RUN: %clang -target arm -march=armv9.3-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V93A %s
// CHECK-V93A: "-cc1"{{.*}} "-triple" "armv9.3{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V93A %s
// RUN: %clang -target armv9.3a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V93A %s
// RUN: %clang -target armeb -march=armebv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V93A %s
// RUN: %clang -target armeb -march=armebv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V93A %s
// RUN: %clang -target arm -march=armebv9.3a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V93A %s
// RUN: %clang -target arm -march=armebv9.3-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V93A %s
// CHECK-BE-V93A: "-cc1"{{.*}} "-triple" "armebv9.3{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv9.4a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V94A %s
// RUN: %clang -target arm -march=armv9.4a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V94A %s
// RUN: %clang -target arm -march=armv9.4-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V94A %s
// RUN: %clang -target arm -march=armv9.4a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V94A %s
// RUN: %clang -target armv9.4a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V94A %s
// RUN: %clang -target arm -march=armv9.4a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V94A %s
// RUN: %clang -target arm -mlittle-endian -march=armv9.4-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V94A %s
// CHECK-V94A: "-cc1"{{.*}} "-triple" "armv9.4{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv9.4a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V94A %s
// RUN: %clang -target armv9.4a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V94A %s
// RUN: %clang -target armeb -march=armebv9.4a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V94A %s
// RUN: %clang -target armeb -march=armebv9.4-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V94A %s
// RUN: %clang -target arm -march=armebv9.4a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V94A %s
// RUN: %clang -target arm -march=armebv9.4-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V94A %s
// CHECK-BE-V94A: "-cc1"{{.*}} "-triple" "armebv9.4{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V95A %s
// RUN: %clang -target arm -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V95A %s
// RUN: %clang -target arm -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V95A %s
// RUN: %clang -target arm -march=armv9.5a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V95A %s
// RUN: %clang -target armv9.5a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V95A %s
// RUN: %clang -target arm -march=armv9.5a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V95A %s
// RUN: %clang -target arm -mlittle-endian -march=armv9.5-a -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V95A %s
// CHECK-V95A: "-cc1"{{.*}} "-triple" "armv9.5{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target armebv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V95A %s
// RUN: %clang -target armv9.5a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V95A %s
// RUN: %clang -target armeb -march=armebv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V95A %s
// RUN: %clang -target armeb -march=armebv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V95A %s
// RUN: %clang -target arm -march=armebv9.5a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V95A %s
// RUN: %clang -target arm -march=armebv9.5-a -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-BE-V95A %s
// CHECK-BE-V95A: "-cc1"{{.*}} "-triple" "armebv9.5{{.*}}" "-target-cpu" "generic"
