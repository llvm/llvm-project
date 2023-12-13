# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func2-gcs.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3-gcs.s -o %t3.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func2.s -o %t2no.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3.s -o %t3no.o

# GCS should be enabled when it's enabled in all inputs or when it's forced on.

# RUN: ld.lld %t1.o %t2.o %t3.o --shared -o %t.exe
# RUN: llvm-readelf -n %t.exe | FileCheck --check-prefix GCS %s
# RUN: ld.lld %t1.o %t3.o --shared -o %t.so
# RUN: llvm-readelf -n %t.so | FileCheck --check-prefix GCS %s
# RUN: ld.lld %t1.o %t2no.o %t3.o --shared -o %tforce.exe -z gcs
# RUN: llvm-readelf -n %tforce.exe | FileCheck --allow-empty --check-prefix GCS %s
# RUN: ld.lld %t2.o %t3no.o --shared -o %tforce.so -z gcs=always
# RUN: llvm-readelf -n %tforce.so | FileCheck --allow-empty --check-prefix GCS %s
# RUN: ld.lld %t2.o %t3no.o --shared -o %tforce2.so -z gcs=never -z gcs=always
# RUN: llvm-readelf -n %tforce2.so | FileCheck --allow-empty --check-prefix GCS %s

# GCS: Properties:    aarch64 feature: GCS

# GCS should not be enabled if it's not enabled in at least one input, and we
# should warn or error when using the report option.

# RUN: ld.lld %t1.o %t2no.o %t3.o --shared -o %tno.exe
# RUN: llvm-readelf -n %tno.exe | FileCheck --allow-empty --check-prefix NOGCS %s
# RUN: ld.lld %t2.o %t3no.o --shared -o %tno.so
# RUN: llvm-readelf -n %tno.so | FileCheck --allow-empty --check-prefix NOGCS %s
# RUN: ld.lld %t1.o %t2no.o %t3.o --shared -o %tno.exe -z gcs-report=warning 2>&1 | FileCheck --check-prefix=REPORT-WARN %s
# RUN: llvm-readelf -n %tno.exe | FileCheck --allow-empty --check-prefix NOGCS %s
# RUN: not ld.lld %t2.o %t3no.o --shared -o %tno.so -z gcs-report=error 2>&1 | FileCheck --check-prefix=REPORT-ERROR %s

# NOGCS-NOT: Properties
# REPORT-WARN: warning: {{.*}}tmp2no.o: -z gcs-report: file does not have GNU_PROPERTY_AARCH64_FEATURE_1_GCS property
# REPORT-ERROR: error: {{.*}}tmp3no.o: -z gcs-report: file does not have GNU_PROPERTY_AARCH64_FEATURE_1_GCS property

# GCS should be disabled with gcs=never, even if GCS is present in all inputs.

# RUN: ld.lld %t1.o %t2.o %t3.o -z gcs=never --shared -o %tnever.exe
# RUN: llvm-readelf -n %tnever.exe | FileCheck --allow-empty --check-prefix NOGCS %s
# RUN: ld.lld %t1.o %t2.o %t3.o -z gcs=always -z gcs=never --shared -o %tnever2.exe
# RUN: llvm-readelf -n %tnever2.exe | FileCheck --allow-empty --check-prefix NOGCS %s

# An invalid gcs option should give an error
# RUN: not ld.lld %t1.o %t2.o %t3.o -z gcs=nonsense -o %tinvalid.exe 2>&1 | FileCheck --check-prefix=INVALID %s

# INVALID: error: unknown -z gcs= value: nonsense

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
