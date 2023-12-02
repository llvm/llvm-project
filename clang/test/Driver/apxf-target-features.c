// RUN: %clang -target x86_64-unknown-linux-gnu -mapxf %s -### -o %t.o 2>&1 | FileCheck -check-prefix=APXF %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mno-apxf %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-APXF %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mno-apxf -mapxf %s -### -o %t.o 2>&1 | FileCheck -check-prefix=APXF %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapxf -mno-apxf %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-APXF %s
//
// APXF: "-target-feature" "+egpr" "-target-feature" "+push2pop2" "-target-feature" "+ppx" "-target-feature" "+ndd" "-target-feature" "+ccmp" "-target-feature" "+cf"
// NO-APXF: "-target-feature" "-egpr" "-target-feature" "-push2pop2" "-target-feature" "-ppx" "-target-feature" "-ndd" "-target-feature" "-ccmp" "-target-feature" "-cf"

// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=egpr,ndd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=EGPR-NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=egpr -mapx-features=ndd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=EGPR-NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mno-apx-features=egpr -mno-apx-features=ndd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-EGPR-NO-NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mno-apx-features=egpr -mapx-features=egpr,ndd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=EGPR-NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mno-apx-features=egpr,ndd -mapx-features=egpr %s -### -o %t.o 2>&1 | FileCheck -check-prefix=EGPR-NO-NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=egpr,ndd -mno-apx-features=egpr %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-EGPR-NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=egpr -mno-apx-features=egpr,ndd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-EGPR-NO-NDD %s
//
// EGPR-NDD: "-target-feature" "+egpr" "-target-feature" "+ndd"
// EGPR-NO-NDD: "-target-feature" "-ndd" "-target-feature" "+egpr"
// NO-EGPR-NDD: "-target-feature" "+ndd" "-target-feature" "-egpr"
// NO-EGPR-NO-NDD: "-target-feature" "-egpr" "-target-feature" "-ndd"

// RUN: not %clang -target x86_64-unknown-linux-gnu -mapx-features=egpr,foo,bar %s -### -o %t.o 2>&1 | FileCheck -check-prefix=ERROR %s
//
// ERROR: unsupported argument 'foo' to option '-mapx-features='
// ERROR: unsupported argument 'bar' to option '-mapx-features='
