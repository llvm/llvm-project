// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=full -funified-lto 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin -funified-lto 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang --target=x86_64-scei-ps4 -### %s -flto=full -funified-lto 2>&1 | FileCheck --check-prefix=UNIT %s
// RUN: %clang --target=x86_64-scei-ps4 -### %s -flto=thin -funified-lto 2>&1 | FileCheck --check-prefix=NOUNIT %s

// UNIT: "-flto-unit"
// NOUNIT-NOT: "-flto-unit"

// RUN: %clang --target=x86_64-sie-ps5 -### %s -funified-lto 2>&1 | FileCheck --check-prefix=NOUNILTO %s
// NOUNILTO: "-cc1"
// NOUNILTO-NOT: "-funified-lto

// On PlayStation -funified-lto is the default. `-flto(=...)` influences the
// `--lto=...` option passed to linker, unless `-fno-unified-lto` is supplied.
// PS4:
// RUN: %clang --target=x86_64-sie-ps4 -### %s 2>&1 | FileCheck --check-prefixes=LD,LTOFULL %s
// RUN: %clang --target=x86_64-sie-ps4 -### %s -flto 2>&1 | FileCheck --check-prefixes=LD,LTOFULL %s
// RUN: %clang --target=x86_64-sie-ps4 -### %s -flto=full 2>&1 | FileCheck --check-prefixes=LD,LTOFULL %s
// RUN: %clang --target=x86_64-sie-ps4 -### %s -flto=thin 2>&1 | FileCheck --check-prefixes=LD,LTOTHIN %s
// RUN: %clang --target=x86_64-sie-ps4 -### %s -fno-unified-lto -flto=full 2>&1 | FileCheck --check-prefixes=LD,NOLTO %s
// RUN: %clang --target=x86_64-sie-ps4 -### %s -fno-unified-lto -flto=thin 2>&1 | FileCheck --check-prefixes=LD,NOLTO %s
// PS5:
// RUN: %clang --target=x86_64-sie-ps5 -### %s 2>&1 | FileCheck --check-prefixes=LD,LTOFULL %s
// RUN: %clang --target=x86_64-sie-ps5 -### %s -flto 2>&1 | FileCheck --check-prefixes=LD,LTOFULL %s
// RUN: %clang --target=x86_64-sie-ps5 -### %s -flto=full 2>&1 | FileCheck --check-prefixes=LD,LTOFULL %s
// RUN: %clang --target=x86_64-sie-ps5 -### %s -flto=thin 2>&1 | FileCheck --check-prefixes=LD,LTOTHIN %s
// RUN: %clang --target=x86_64-sie-ps5 -### %s -fno-unified-lto -flto=full 2>&1 | FileCheck --check-prefixes=LD,NOLTO %s
// RUN: %clang --target=x86_64-sie-ps5 -### %s -fno-unified-lto -flto=thin 2>&1 | FileCheck --check-prefixes=LD,NOLTO %s

// LD: {{.*ld(\.exe)?}}"
// LTOFULL-SAME: "--lto=full"
// LTOTHIN-SAME: "--lto=thin"
// NOLTO-NOT: "--lto
