// Check that types, widths, __GCC_ATOMIC* macros, etc. match on the host and
// device sides of CUDA compilations.  Note that we filter out long double, as
// this is intentionally different on host and device.
//
// FIXME: We really should make __GCC_HAVE_SYNC_COMPARE_AND_SWAP identical on
// host and device, but architecturally this is difficult at the moment.

// RUN: %clang --cuda-host-only -nocudainc -target i386-unknown-linux-gnu -x cuda -E -dM -o - /dev/null \
// RUN:   | grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF|WIDTH\)\|define __GCC_ATOMIC' \
// RUN:   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/i386-host-defines-filtered
// RUN: %clang --cuda-device-only -nocudainc -nocudalib -target i386-unknown-linux-gnu -x cuda -E -dM -o - /dev/null \
// RUN:   | grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF|WIDTH\)\|define __GCC_ATOMIC' \
// RUN:   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/i386-device-defines-filtered
// RUN: diff %T/i386-host-defines-filtered %T/i386-device-defines-filtered

// RUN: %clang --cuda-host-only -nocudainc -target x86_64-unknown-linux-gnu -x cuda -E -dM -o - /dev/null \
// RUN:   | grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF|WIDTH\)\|define __GCC_ATOMIC' \
// RUN:   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/x86_64-host-defines-filtered
// RUN: %clang --cuda-device-only -nocudainc -nocudalib -target x86_64-unknown-linux-gnu -x cuda -E -dM -o - /dev/null \
// RUN:   | grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF|WIDTH\)\|define __GCC_ATOMIC' \
// RUN:   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/x86_64-device-defines-filtered
// RUN: diff %T/x86_64-host-defines-filtered %T/x86_64-device-defines-filtered

// RUN: %clang --cuda-host-only -nocudainc -target powerpc64-unknown-linux-gnu -x cuda -E -dM -o - /dev/null \
// RUN:   | grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF|WIDTH\)\|define __GCC_ATOMIC' \
// RUN:   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/powerpc64-host-defines-filtered
// RUN: %clang --cuda-device-only -nocudainc -nocudalib -target powerpc64-unknown-linux-gnu -x cuda -E -dM -o - /dev/null \
// RUN:   | grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF|WIDTH\)\|define __GCC_ATOMIC' \
// RUN:   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/powerpc64-device-defines-filtered
// RUN: diff %T/powerpc64-host-defines-filtered %T/powerpc64-device-defines-filtered

// RUN: %clang --cuda-host-only -nocudainc -target i386-windows-msvc -x cuda -E -dM -o - /dev/null \
// RUN:   | grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF|WIDTH\)\|define __GCC_ATOMIC' \
// RUN:   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/i386-msvc-host-defines-filtered
// RUN: %clang --cuda-device-only -nocudainc -nocudalib -target i386-windows-msvc -x cuda -E -dM -o - /dev/null \
// RUN:   | grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF|WIDTH\)\|define __GCC_ATOMIC' \
// RUN:   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/i386-msvc-device-defines-filtered
// RUN: diff %T/i386-msvc-host-defines-filtered %T/i386-msvc-device-defines-filtered

// RUN: %clang --cuda-host-only -nocudainc -target x86_64-windows-msvc -x cuda -E -dM -o - /dev/null \
// RUN:   | grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF|WIDTH\)\|define __GCC_ATOMIC' \
// RUN:   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/x86_64-msvc-host-defines-filtered
// RUN: %clang --cuda-device-only -nocudainc -nocudalib -target x86_64-windows-msvc -x cuda -E -dM -o - /dev/null \
// RUN:   | grep 'define __[^ ]*\(TYPE\|MAX\|SIZEOF|WIDTH\)\|define __GCC_ATOMIC' \
// RUN:   | grep -v '__LDBL\|_LONG_DOUBLE' > %T/x86_64-msvc-device-defines-filtered
// RUN: diff %T/x86_64-msvc-host-defines-filtered %T/x86_64-msvc-device-defines-filtered
