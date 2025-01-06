# RUN: not %no-fatal-warnings-lld -o /dev/null --irpgo-profile-sort %s --call-graph-profile-sort 2>&1 | FileCheck %s --check-prefix=IRPGO-ERR
# RUN: not %no-fatal-warnings-lld -o /dev/null --irpgo-profile-sort=%s --call-graph-profile-sort 2>&1 | FileCheck %s --check-prefix=IRPGO-ERR
# IRPGO-ERR: --irpgo-profile-sort is incompatible with --call-graph-profile-sort

# RUN: not %lld -o /dev/null --compression-sort=function --call-graph-profile-sort %s 2>&1 | FileCheck %s --check-prefix=COMPRESSION-ERR
# RUN: not %lld -o /dev/null --bp-compression-sort=function --call-graph-profile-sort %s 2>&1 | FileCheck %s --check-prefix=COMPRESSION-ERR
# COMPRESSION-ERR: --bp-compression-sort= is incompatible with --call-graph-profile-sort

# RUN: not %lld -o /dev/null --compression-sort=malformed 2>&1 | FileCheck %s --check-prefix=COMPRESSION-MALFORM
# RUN: not %lld -o /dev/null --bp-compression-sort=malformed 2>&1 | FileCheck %s --check-prefix=COMPRESSION-MALFORM
# COMPRESSION-MALFORM: unknown value `malformed` for --bp-compression-sort=

# RUN: not %lld -o /dev/null --compression-sort-startup-functions 2>&1 | FileCheck %s --check-prefix=STARTUP
# RUN: not %lld -o /dev/null --bp-compression-sort-startup-functions 2>&1 | FileCheck %s --check-prefix=STARTUP
# STARTUP: --bp-compression-sort-startup-functions must be used with --bp-startup-sort=function

# RUN: not %lld -o /dev/null --irpgo-profile %s --bp-startup-sort=function --call-graph-profile-sort 2>&1 | FileCheck %s --check-prefix=IRPGO-STARTUP
# RUN: not %lld -o /dev/null --irpgo-profile=%s --bp-startup-sort=function --call-graph-profile-sort 2>&1 | FileCheck %s --check-prefix=IRPGO-STARTUP
# IRPGO-STARTUP: --bp-startup-sort= is incompatible with --call-graph-profile-sort

# RUN: not %lld -o /dev/null --bp-startup-sort=function 2>&1 | FileCheck %s --check-prefix=STARTUP-COMPRESSION
# STARTUP-COMPRESSION: --bp-startup-sort=function must be used with --irpgo-profile
