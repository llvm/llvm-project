! XFAIL: *
! Test that Clang can forward all of the flags which are documented as
! being supported by gfortran to GCC when falling back to GCC for
! a fortran input file.
!
! RUN: %clang -no-canonical-prefixes -target i386-linux -### %s -o %t 2>&1 \
! RUN:     -Aquestion=answer \
! RUN:     -A-question=answer \
! RUN:     -C \
! RUN:     -CC \
! RUN:     -Dmacro \
! RUN:     -Dmacro=value \
! RUN:     -H \
! RUN:     -Isome/directory \
! RUN:     -Jsome/other/directory \
! RUN:     -P \
! RUN:     -Umacro \
! RUN:     -Waliasing \
! RUN:     -Walign-commons \
! RUN:     -Wall \
! RUN:     -Wampersand \
! RUN:     -Warray-bounds \
! RUN:     -Wc-binding-type \
! RUN:     -Wcharacter-truncation \
! RUN:     -Wconversion \
! RUN:     -Wfunction-elimination \
! RUN:     -Wimplicit-interface \
! RUN:     -Wimplicit-procedure \
! RUN:     -Wintrinsic-shadow \
! RUN:     -Wintrinsics-std \
! RUN:     -Wline-truncation \
! RUN:     -Wreal-q-constant \
! RUN:     -Wrealloc-lhs \
! RUN:     -Wsurprising \
! RUN:     -Wtabs \
! RUN:     -Wtarget-lifetime \
! RUN:     -Wunderflow \
! RUN:     -Wunused-parameter \
! RUN:     -cpp \
! RUN:     -dD \
! RUN:     -dI \
! RUN:     -dM \
! RUN:     -dN \
! RUN:     -dU \
! RUN:     -faggressive-function-elimination \
! RUN:     -falign-commons \
! RUN:     -fall-intrinsics \
! RUN:     -fautomatic \
! RUN:     -fbackslash \
! RUN:     -fbacktrace \
! RUN:     -fblas-matmul-limit=42 \
! RUN:     -fbounds-check \
! RUN:     -fcheck-array-temporaries \
! RUN:     -fcheck=all \
! RUN:     -fcoarray=none \
! RUN:     -fconvert=foobar \
! RUN:     -fcray-pointer \
! RUN:     -fd-lines-as-code \
! RUN:     -fd-lines-as-comments \
! RUN:     -fdefault-double-8 \
! RUN:     -fdefault-integer-8 \
! RUN:     -fdefault-real-8 \
! RUN:     -fdollar-ok \
! RUN:     -fdump-fortran-optimized \
! RUN:     -fdump-fortran-original \
! RUN:     -fdump-parse-tree \
! RUN:     -fexternal-blas \
! RUN:     -ff2c \
! RUN:     -ffixed-form \
! RUN:     -ffixed-line-length-42 \
! RUN:     -ffpe-trap=list \
! RUN:     -ffree-form \
! RUN:     -ffree-line-length-42 \
! RUN:     -ffrontend-optimize \
! RUN:     -fimplicit-none \
! RUN:     -finit-character=n \
! RUN:     -finit-integer=n \
! RUN:     -finit-local-zero \
! RUN:     -finit-logical=false \
! RUN:     -finit-real=zero \
! RUN:     -finteger-4-integer-8 \
! RUN:     -fintrinsic-modules-path \
! RUN:     -fmax-array-constructor=42 \
! RUN:     -fmax-errors=42 \
! RUN:     -fmax-identifier-length \
! RUN:     -fmax-stack-var-size=42 \
! RUN:     -fmax-subrecord-length=42 \
! RUN:     -fmodule-private \
! RUN:     -fopenmp \
! RUN:     -fpack-derived \
! RUN:     -fprotect-parens \
! RUN:     -frange-check \
! RUN:     -freal-4-real-10 \
! RUN:     -freal-4-real-16 \
! RUN:     -freal-4-real-8 \
! RUN:     -freal-8-real-10 \
! RUN:     -freal-8-real-16 \
! RUN:     -freal-8-real-4 \
! RUN:     -frealloc-lhs \
! RUN:     -frecord-marker=42 \
! RUN:     -frecursive \
! RUN:     -frepack-arrays \
! RUN:     -fsecond-underscore \
! RUN:     -fshort-enums \
! RUN:     -fsign-zero \
! RUN:     -fstack-arrays \
! RUN:     -fsyntax-only \
! RUN:     -funderscoring \
! RUN:     -fwhole-file \
! RUN:     -imultilib \
! RUN:     -iprefix \
! RUN:     -iquote \
! RUN:     -isysroot \
! RUN:     -isystem \
! RUN:     -nocpp \
! RUN:     -nostdinc \
! RUN:     -pedantic \
! RUN:     -pedantic-errors \
! RUN:     -static-libgfortran \
! RUN:     -std=f90 \
! RUN:     -undef \
! RUN:   | FileCheck %s
!
! FIXME: Several of these shouldn't necessarily be rendered separately
! when passing to GCC... Hopefully their driver handles this.
!
! CHECK: "-Aquestion=answer"
! CHECK: "-A-question=answer"
! CHECK: "-C"
! CHECK: "-CC"
! CHECK: "-D" "macro"
! CHECK: "-D" "macro=value"
! CHECK: "-H"
! CHECK: "-I" "some/directory"
! CHECK: "-Jsome/other/directory"
! CHECK: "-P"
! CHECK: "-U" "macro"
! CHECK: "-Waliasing"
! CHECK: "-Walign-commons"
! CHECK: "-Wall"
! CHECK: "-Wampersand"
! CHECK: "-Warray-bounds"
! CHECK: "-Wc-binding-type"
! CHECK: "-Wcharacter-truncation"
! CHECK: "-Wconversion"
! CHECK: "-Wfunction-elimination"
! CHECK: "-Wimplicit-interface"
! CHECK: "-Wimplicit-procedure"
! CHECK: "-Wintrinsic-shadow"
! CHECK: "-Wintrinsics-std"
! CHECK: "-Wline-truncation"
! CHECK: "-Wreal-q-constant"
! CHECK: "-Wrealloc-lhs"
! CHECK: "-Wsurprising"
! CHECK: "-Wtabs"
! CHECK: "-Wtarget-lifetime"
! CHECK: "-Wunderflow"
! CHECK: "-Wunused-parameter"
! CHECK: "-cpp"
! CHECK: "-dD"
! CHECK: "-dI"
! CHECK: "-dM"
! CHECK: "-dN"
! CHECK: "-dU"
! CHECK: "-faggressive-function-elimination"
! CHECK: "-falign-commons"
! CHECK: "-fall-intrinsics"
! CHECK: "-fautomatic"
! CHECK: "-fbackslash"
! CHECK: "-fbacktrace"
! CHECK: "-fblas-matmul-limit=42"
! CHECK: "-fbounds-check"
! CHECK: "-fcheck-array-temporaries"
! CHECK: "-fcheck=all"
! CHECK: "-fcoarray=none"
! CHECK: "-fconvert=foobar"
! CHECK: "-fcray-pointer"
! CHECK: "-fd-lines-as-code"
! CHECK: "-fd-lines-as-comments"
! CHECK: "-fdefault-double-8"
! CHECK: "-fdefault-integer-8"
! CHECK: "-fdefault-real-8"
! CHECK: "-fdollar-ok"
! CHECK: "-fdump-fortran-optimized"
! CHECK: "-fdump-fortran-original"
! CHECK: "-fdump-parse-tree"
! CHECK: "-fexternal-blas"
! CHECK: "-ff2c"
! CHECK: "-ffixed-form"
! CHECK: "-ffixed-line-length-42"
! CHECK: "-ffpe-trap=list"
! CHECK: "-ffree-form"
! CHECK: "-ffree-line-length-42"
! CHECK: "-ffrontend-optimize"
! CHECK: "-fimplicit-none"
! CHECK: "-finit-character=n"
! CHECK: "-finit-integer=n"
! CHECK: "-finit-local-zero"
! CHECK: "-finit-logical=false"
! CHECK: "-finit-real=zero"
! CHECK: "-finteger-4-integer-8"
! CHECK: "-fintrinsic-modules-path"
! CHECK: "-fmax-array-constructor=42"
! CHECK: "-fmax-errors=42"
! CHECK: "-fmax-identifier-length"
! CHECK: "-fmax-stack-var-size=42"
! CHECK: "-fmax-subrecord-length=42"
! CHECK: "-fmodule-private"
! CHECK: "-fopenmp"
! CHECK: "-fpack-derived"
! CHECK: "-fprotect-parens"
! CHECK: "-frange-check"
! CHECK: "-freal-4-real-10"
! CHECK: "-freal-4-real-16"
! CHECK: "-freal-4-real-8"
! CHECK: "-freal-8-real-10"
! CHECK: "-freal-8-real-16"
! CHECK: "-freal-8-real-4"
! CHECK: "-frealloc-lhs"
! CHECK: "-frecord-marker=42"
! CHECK: "-frecursive"
! CHECK: "-frepack-arrays"
! CHECK: "-fsecond-underscore"
! CHECK: "-fshort-enums"
! CHECK: "-fsign-zero"
! CHECK: "-fstack-arrays"
! CHECK: "-funderscoring"
! CHECK: "-fwhole-file"
! CHECK: "-imultilib"
! CHECK: "-iprefix"
! CHECK: "-iquote"
! CHECK: "-isysroot"
! CHECK: "-isystem"
! CHECK: "-nocpp"
! CHECK: "-nostdinc"
! CHECK: "-pedantic"
! CHECK: "-pedantic-errors"
! CHECK: "-static-libgfortran"
! CHECK: "-std=f90"
! CHECK: "-undef"
!
! Clang understands this one and orders it weirdly.
! CHECK: "-fsyntax-only"
!
! PR22234: Ensure that -fsyntax-only doesn't complain about output types and
!          passes along correctly.
! RUN: %clang -no-canonical-prefixes -target i386-linux -fsyntax-only -### %s -o %t 2>&1 | \
! grep for error message and command-line
! RUN: grep -e error: -e -fsyntax-only | FileCheck %s --check-prefix=CHECK-PR22234
!
! CHECK-PR22234-NOT: clang: error: invalid output type
! CHECK-PR22234: "-fsyntax-only"
!
! Regression test for the bug introduced with PR22234 fix.
! Make sure -fsyntax-only is not passed to gfortran during normal compilation.
!
! RUN: %clang -no-canonical-prefixes -target i386-linux -### %s -o %t 2>&1 \
! RUN: | FileCheck %s --check-prefix=CHECK-PR22234-R
! CHECK-PR22234-R-NOT: "-fsyntax-only"
