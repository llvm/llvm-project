# REQUIRES: aarch64
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu func1.s -o func1.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu func2.s -o func2.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu func2-gcs.s -o func2-gcs.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu func3.s -o func3.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu func3-gcs.s -o func3-gcs.o

## GCS should be enabled when it's enabled in all inputs or when it's forced on.

# RUN: ld.lld func1.o func2-gcs.o func3-gcs.o --shared -o gcs.exe
# RUN: llvm-readelf -n gcs.exe | FileCheck --check-prefix GCS %s
# RUN: ld.lld func1.o func3-gcs.o --shared -o gcs.so
# RUN: llvm-readelf -n gcs.so | FileCheck --check-prefix GCS %s
# RUN: ld.lld func1.o func2.o func3-gcs.o --shared -o force-gcs.exe -z gcs
# RUN: llvm-readelf -n force-gcs.exe | FileCheck --allow-empty --check-prefix GCS %s
# RUN: ld.lld func2-gcs.o func3.o --shared -o force-gcs.so -z gcs=always
# RUN: llvm-readelf -n force-gcs.so | FileCheck --allow-empty --check-prefix GCS %s
# RUN: ld.lld func2-gcs.o func3.o --shared -o force-gcs2.so -z gcs=never -z gcs=always
# RUN: llvm-readelf -n force-gcs2.so | FileCheck --allow-empty --check-prefix GCS %s

# GCS: Properties:    aarch64 feature: GCS

## GCS should not be enabled if it's not enabled in at least one input, and we
## should warn or error when using the report option.

# RUN: ld.lld func1.o func2.o func3-gcs.o --shared -o no-gcs.exe
# RUN: llvm-readelf -n no-gcs.exe | FileCheck --allow-empty --check-prefix NOGCS %s
# RUN: ld.lld func2-gcs.o func3.o --shared -o no-gcs.so
# RUN: llvm-readelf -n no-gcs.so | FileCheck --allow-empty --check-prefix NOGCS %s
# RUN: ld.lld func1.o func2.o func3-gcs.o --shared -o no-gcs.exe -z gcs-report=warning 2>&1 | FileCheck --check-prefix=REPORT-WARN %s
# RUN: llvm-readelf -n no-gcs.exe | FileCheck --allow-empty --check-prefix NOGCS %s
# RUN: not ld.lld func2-gcs.o func3.o --shared -o no-gcs.so -z gcs-report=error 2>&1 | FileCheck --check-prefix=REPORT-ERROR %s

# NOGCS-NOT: Properties
# REPORT-WARN: warning: func2.o: -z gcs-report: file does not have GNU_PROPERTY_AARCH64_FEATURE_1_GCS property
# REPORT-ERROR: error: func3.o: -z gcs-report: file does not have GNU_PROPERTY_AARCH64_FEATURE_1_GCS property

## GCS should be disabled with gcs=never, even if GCS is present in all inputs.

# RUN: ld.lld func1.o func2-gcs.o func3-gcs.o -z gcs=never --shared -o never-gcs.exe
# RUN: llvm-readelf -n never-gcs.exe | FileCheck --allow-empty --check-prefix NOGCS %s
# RUN: ld.lld func1.o func2-gcs.o func3-gcs.o -z gcs=always -z gcs=never --shared -o never-gcs2.exe
# RUN: llvm-readelf -n never-gcs2.exe | FileCheck --allow-empty --check-prefix NOGCS %s

## An invalid gcs option should give an error
# RUN: not ld.lld func1.o func2-gcs.o func3-gcs.o -z gcs=nonsense -o invalid.exe 2>&1 | FileCheck --check-prefix=INVALID %s

# INVALID: error: unknown -z gcs= value: nonsense

#--- func1.s
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
.type func1,%function
func1:
  bl func2
  ret

#--- func2.s

.text
.globl func2
.type func2,@function
func2:
  .globl func3
  .type func3, @function
  bl func3
  ret

#--- func2-gcs.s

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
.globl func2
.type func2,@function
func2:
  .globl func3
  .type func3, @function
  bl func3
  ret

#--- func3.s

.text
.globl func3
.type func3,@function
func3:
  ret

#--- func3-gcs.s

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
.globl func3
.type func3,@function
func3:
  ret
