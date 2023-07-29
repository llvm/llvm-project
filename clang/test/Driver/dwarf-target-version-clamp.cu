// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// Verify that DWARF version is properly clamped for nvptx, but not for the host.
// RUN: %clang -### -target x86_64-linux-gnu -c %s -gdwarf-5 -gembed-source 2>&1 \
// RUN: | FileCheck %s --check-prefix=DWARF-CLAMP
// RUN: %clang -### -target x86_64-linux-gnu -c %s -ggdb -gembed-source -gdwarf-5 2>&1 \
// RUN: | FileCheck %s --check-prefix=DWARF-CLAMP

// DWARF-CLAMP: "-triple" "nvptx64-nvidia-cuda"
// DWARF-CLAMP-SAME: -dwarf-version=2
// DWARF-CLAMP: "-triple" "x86_64
// DWARF-CLAMP-SAME: -dwarf-version=5
