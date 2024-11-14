# RUN: not %lld -o /dev/null --irpgo-profile-sort %s --call-graph-profile-sort 2>&1 | FileCheck %s --check-prefix=IRPGO-ERR
# RUN: not %lld -o /dev/null --irpgo-profile-sort=%s --call-graph-profile-sort 2>&1 | FileCheck %s --check-prefix=IRPGO-ERR
# IRPGO-ERR: --irpgo-profile-sort is incompatible with --call-graph-profile-sort

# RUN: not %lld -o /dev/null --compression-sort=function --call-graph-profile-sort %s 2>&1 | FileCheck %s --check-prefix=COMPRESSION-ERR
# COMPRESSION-ERR: --compression-sort= is incompatible with --call-graph-profile-sort

# RUN: not %lld -o /dev/null --compression-sort=malformed 2>&1 | FileCheck %s --check-prefix=COMPRESSION-MALFORM
# COMPRESSION-MALFORM: unknown value `malformed` for --compression-sort=
