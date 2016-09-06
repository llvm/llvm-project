// Check that types, widths, etc. match on the host and device sides of CUDA
// compilations.  Note that we filter out long double, as this is intentionally
// different on host and device.

// RUN: %clang --cuda-host-only -nocudainc -target i386-unknown-linux-gnu -x cuda -E -dM -o - /dev/null > %T/i386-host-defines
// RUN: %clang --cuda-device-only -nocudainc -nocudalib -target i386-unknown-linux-gnu -x cuda -E -dM -o - /dev/null > %T/i386-device-defines
// RUN: grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF|WIDTH\)' %T/i386-host-defines   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/i386-host-defines-filtered
// RUN: grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF|WIDTH\)' %T/i386-device-defines | grep -v '__LDBL\|_LONG_DOUBLE' > %T/i386-device-defines-filtered
// RUN: diff %T/i386-host-defines-filtered %T/i386-device-defines-filtered

// RUN: %clang --cuda-host-only -nocudainc -target x86_64-unknown-linux-gnu -x cuda -E -dM -o - /dev/null > %T/x86_64-host-defines
// RUN: %clang --cuda-device-only -nocudainc -nocudalib -target x86_64-unknown-linux-gnu -x cuda -E -dM -o - /dev/null > %T/x86_64-device-defines
// RUN: grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF\|WIDTH\)' %T/x86_64-host-defines   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/x86_64-host-defines-filtered
// RUN: grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF\|WIDTH\)' %T/x86_64-device-defines | grep -v '__LDBL\|_LONG_DOUBLE' > %T/x86_64-device-defines-filtered
// RUN: diff %T/x86_64-host-defines-filtered %T/x86_64-device-defines-filtered

// RUN: %clang --cuda-host-only -nocudainc -target powerpc64-unknown-linux-gnu -x cuda -E -dM -o - /dev/null > %T/powerpc64-host-defines
// RUN: %clang --cuda-device-only -nocudainc -nocudalib -target powerpc64-unknown-linux-gnu -x cuda -E -dM -o - /dev/null > %T/powerpc64-device-defines
// RUN: grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF\|WIDTH\)' %T/powerpc64-host-defines   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/powerpc64-host-defines-filtered
// RUN: grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF\|WIDTH\)' %T/powerpc64-device-defines | grep -v '__LDBL\|_LONG_DOUBLE' > %T/powerpc64-device-defines-filtered
// RUN: diff %T/powerpc64-host-defines-filtered %T/powerpc64-device-defines-filtered
