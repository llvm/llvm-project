# REQUIRES: aarch64
## Test the Guarded Control Stack (GCS) feature.
## Naming convention: *-s.s files enable GCS.
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 f1-s.s -o f1-s.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 f2.s -o f2.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 f2-s.s -o f2-s.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 f3.s -o f3.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 f3-s.s -o f3-s.o

## GCS should be enabled when it's enabled in all inputs or when it's forced on.

# RUN: ld.lld f1-s.o f2-s.o f3-s.o -o out --fatal-warnings
# RUN: llvm-readelf -n out | FileCheck --check-prefix GCS %s
# RUN: ld.lld f1-s.o f2.o f3-s.o -o out.force -z gcs=always --fatal-warnings
# RUN: llvm-readelf -n out.force | FileCheck --check-prefix GCS %s
# RUN: ld.lld f2-s.o f3.o --shared -o out.force.so -z gcs=never -z gcs=always --fatal-warnings
# RUN: llvm-readelf -n out.force.so | FileCheck --check-prefix GCS %s

# GCS: Properties:    aarch64 feature: GCS

## GCS should not be enabled if it's not enabled in at least one input.

# RUN: ld.lld f1-s.o f2.o f3-s.o -o out.no --fatal-warnings
# RUN: llvm-readelf -n out.no | count 0
# RUN: ld.lld f2-s.o f3.o --shared -o out.no.so

## GCS should be disabled with gcs=never, even if GCS is present in all inputs.

# RUN: ld.lld f1-s.o f2-s.o f3-s.o -z gcs=always -z gcs=never -o out.never --fatal-warnings
# RUN: llvm-readelf -n out.never | count 0

## gcs-report should report any input files that don't have the gcs property.

# RUN: ld.lld f1-s.o f2.o f3-s.o -z gcs-report=warning 2>&1 | FileCheck --check-prefix=REPORT-WARN %s
# RUN: ld.lld f1-s.o f2.o f3-s.o -z gcs-report=warning -z gcs=always 2>&1 | FileCheck --check-prefix=REPORT-WARN %s
# RUN: ld.lld f1-s.o f2.o f3-s.o -z gcs-report=warning -z gcs=never 2>&1 | FileCheck --check-prefix=REPORT-WARN %s
# RUN: not ld.lld f2-s.o f3.o --shared -z gcs-report=error 2>&1 | FileCheck --check-prefix=REPORT-ERROR %s
# RUN: ld.lld f1-s.o f2-s.o f3-s.o -z gcs-report=warning -z gcs=always 2>&1 | count 0

# REPORT-WARN: warning: f2.o: -z gcs-report: file does not have GNU_PROPERTY_AARCH64_FEATURE_1_GCS property
# REPORT-WARN-NOT: {{.}}
# REPORT-ERROR: error: f3.o: -z gcs-report: file does not have GNU_PROPERTY_AARCH64_FEATURE_1_GCS property
# REPORT-ERROR-NOT: {{.}}

## gcs-report-dynamic should report any dynamic objects that does not have the gcs property. This also ensures the inhertance from gcs-report is working correctly.

# RUN: ld.lld f1-s.o f3-s.o out.no.so out.force.so -z gcs-report=warning -z gcs=always 2>&1 | FileCheck --check-prefix=REPORT-WARN-DYNAMIC %s
# RUN: ld.lld f1-s.o f3-s.o out.no.so out.force.so -z gcs-report=error -z gcs=always 2>&1 | FileCheck --check-prefix=REPORT-WARN-DYNAMIC %s
# RUN: ld.lld f1-s.o f3-s.o out.no.so out.force.so -z gcs-report-dynamic=warning -z gcs=always 2>&1 | FileCheck --check-prefix=REPORT-WARN-DYNAMIC %s
# RUN: not ld.lld f1-s.o f3-s.o out.no.so out.force.so -z gcs-report-dynamic=error -z gcs=always 2>&1 | FileCheck --check-prefix=REPORT-ERROR-DYNAMIC %s
# RUN: ld.lld f1-s.o f3-s.o out.force.so -z gcs-report-dynamic=error -z gcs=always 2>&1 | count 0

# REPORT-WARN-DYNAMIC: warning: out.no.so: GCS is required by -z gcs, but this shared library lacks the necessary property note. The dynamic loader might not enable GCS or refuse to load the program unless all shared library dependencies have the GCS marking.
# REPORT-WARN-DYNAMIC-NOT: {{.}}
# REPORT-ERROR-DYNAMIC: error: out.no.so: GCS is required by -z gcs, but this shared library lacks the necessary property note. The dynamic loader might not enable GCS or refuse to load the program unless all shared library dependencies have the GCS marking.
# REPORT-ERROR-DYNAMIC-NOT: error:

## An invalid gcs option should give an error
# RUN: not ld.lld f1-s.o -z gcs=x -z gcs-report=x -z gcs-report-dynamic=x 2>&1 | FileCheck --check-prefix=INVALID %s

# INVALID: error: unknown -z gcs= value: x
# INVALID: error: unknown -z gcs-report= value: x
# INVALID: error: unknown -z gcs-report-dynamic= value: x

#--- f1-s.s
.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"

.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4
.long 4          // GNU_PROPERTY_AARCH64_FEATURE_1_GCS
.long 0

.text
.globl _start
.type f1,%function
f1:
  bl f2
  ret

#--- f2.s
.text
.globl f2
.type f2,@function
f2:
  .globl f3
  .type f3, @function
  bl f3
  ret

#--- f2-s.s
.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"

.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4
.long 4          // GNU_PROPERTY_AARCH64_FEATURE_1_GCS
.long 0

.text
.globl f2
.type f2,@function
f2:
  .globl f3
  .type f3, @function
  bl f3
  ret

#--- f3.s
.text
.globl f3
.type f3,@function
f3:
  ret

#--- f3-s.s
.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"

.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4
.long 4          // GNU_PROPERTY_AARCH64_FEATURE_1_GCS
.long 0

.text
.globl f3
.type f3,@function
f3:
  ret
