// RUN: %clang -target i686-pc-linux-gnu -### -rtlib=libgcc -nodefaultlibs %s 2>&1 | FileCheck -check-prefix=TEST1 %s
// TEST1-NOT: start-group
// TEST1-NOT: "-lgcc"
// TEST1-NOT: "-lc"
// TEST1: crtbegin
// TEST1: crtend

// RUN: %clang -target i686-pc-linux-gnu -stdlib=libc++ -nodefaultlibs -lstdc++ -### %s 2>&1 | FileCheck -check-prefix=TEST2 %s
// TEST2-NOT: "-lc++"
// TEST2: "-lstdc++"
