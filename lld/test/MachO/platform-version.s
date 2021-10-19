# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

### We test every platform keyword. Sometimes good keywords are coupled
### with bad version strings, so we use *-NOT patterns to ensure that
### no "malformed platform" diagnostic appears in those cases.

# RUN: not %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version \
# RUN:    | FileCheck --check-prefix=FAIL-MISSING %s
# RUN: not %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version wtf \
# RUN:    | FileCheck --check-prefix=FAIL-MISSING %s
# RUN: not %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version lolz 1.2.3.4.5 \
# RUN:    | FileCheck --check-prefix=FAIL-MISSING %s
# FAIL-MISSING: -platform_version: missing argument
# FAIL-MISSING-NOT: malformed platform: {{.*}}
# FAIL-MISSING-NOT: malformed {{minimum|sdk}} version: {{.*}}

# RUN: not %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version macOS -lfoo 2 \
# RUN:     | FileCheck --check-prefix=FAIL-MALFORM %s
# RUN: not %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version iOS 1 2.a \
# RUN:     | FileCheck --check-prefix=FAIL-MALFORM %s
# RUN: not %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version tvOS 1.2.3.4.5 10 \
# RUN:     | FileCheck --check-prefix=FAIL-MALFORM %s
# RUN: not %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version watchOS 10 1.2.3.4.5 \
# RUN:     | FileCheck --check-prefix=FAIL-MALFORM %s
# FAIL-MALFORM-NOT: malformed platform: {{.*}}
# FAIL-MALFORM: malformed {{minimum|sdk}} version: {{.*}}

# RUN: %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version bridgeOS 1 5
# RUN: %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version "Mac Catalyst" 1.2 5.6
# RUN: %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version "iOS Simulator" 1.2.3 5.6.7
# RUN: %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version tvOS-Simulator 1.2.3.4 5.6.7.8
# RUN: %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version watchOS-Simulator 1 5
# RUN: %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version 1 1 5
# RUN: %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version 9 1 5

# RUN: not %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version wtf 1 5 \
# RUN:     | FileCheck --check-prefix=FAIL-PLATFORM %s
# RUN: not %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version 0 1 5 \
# RUN:     | FileCheck --check-prefix=FAIL-PLATFORM %s
# RUN: not %lld -o %t %t.o 2>&1 \
# RUN:        -platform_version 11 1 5 \
# RUN:     | FileCheck --check-prefix=FAIL-PLATFORM %s
# FAIL-PLATFORM: malformed platform: {{.*}}
# FAIL-PLATFORM-NOT: malformed {{minimum|sdk}} version: {{.*}}

.text
.global _main
_main:
  mov $0, %eax
  ret
