// The linker wrapper extracts device code from static libraries by the
// architecture its images are labeled with, so request the canonical name.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -x hip -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx908:xnack+:sramecc- %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CANONICAL
// CANONICAL: "--image=file={{.*}},triple={{[^,]*}},arch=gfx908:sramecc-:xnack+,kind=hip"
// CANONICAL: clang-linker-wrapper{{.*}} "--should-extract=gfx908:sramecc-:xnack+"

/// An architecture removed with --no-offload-arch is not requested.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -x hip -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx906 --offload-arch=gfx90a --no-offload-arch=gfx906 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REMOVED
// REMOVED: clang-linker-wrapper
// REMOVED-NOT: "--should-extract=gfx906"
// REMOVED-SAME: "--should-extract=gfx90a"
// REMOVED-NOT: "--should-extract=gfx906"
