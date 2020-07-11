// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// By default FE assumes -fhip-new-launch-api.

// RUN: %clang -### -target x86_64-unknown-linux-gnu -offload-arch=gfx906 %s \
// RUN:   2>&1 | FileCheck -check-prefixes=NEW %s
// NEW: "-fhip-new-launch-api"

// RUN: %clang -### -target x86_64-unknown-linux-gnu -offload-arch=gfx906 %s \
// RUN:   -fhip-new-launch-api 2>&1 | FileCheck -check-prefixes=NEW %s
// NEW: "-fhip-new-launch-api"

// RUN: %clang -### -target x86_64-unknown-linux-gnu -offload-arch=gfx906 %s \
// RUN:   -fno-hip-new-launch-api 2>&1 | FileCheck -check-prefixes=OLD %s
// OLD-NOT: "-fhip-new-launch-api"
