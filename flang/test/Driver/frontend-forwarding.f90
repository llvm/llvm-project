! Test that flang-new forwards Flang frontend
! options to flang-new -fc1 as expected.

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -finput-charset=utf-8 \
! RUN:     -fdefault-double-8 \
! RUN:     -fdefault-integer-8 \
! RUN:     -fdefault-real-8 \
! RUN:     -flarge-sizes \
! RUN:     -fconvert=little-endian \
! RUN:     -ffp-contract=fast \
! RUN:     -fno-honor-nans \
! RUN:     -fapprox-func \
! RUN:     -fno-signed-zeros \
! RUN:     -fassociative-math \
! RUN:     -freciprocal-math \
! RUN:     -fomit-frame-pointer \
! RUN:     -fpass-plugin=Bye%pluginext \
! RUN:     -fversion-loops-for-stride \
! RUN:     -flang-experimental-hlfir \
! RUN:     -flang-deprecated-no-hlfir \
! RUN:     -fno-ppc-native-vector-element-order \
! RUN:     -fppc-native-vector-element-order \
! RUN:     -mllvm -print-before-all \
! RUN:     -save-temps=obj \
! RUN:     -Rpass \
! RUN:     -Rpass-missed \
! RUN:     -Rpass-analysis \
! RUN:     -Rno-pass \
! RUN:     -Reverything \
! RUN:     -Rno-everything \
! RUN:     -Rpass=inline \
! RUN:     -P \
! RUN:   | FileCheck %s

! CHECK: "-P"
! CHECK: "-finput-charset=utf-8"
! CHECK: "-fdefault-double-8"
! CHECK: "-fdefault-integer-8"
! CHECK: "-fdefault-real-8"
! CHECK: "-flarge-sizes"
! CHECK: "-ffp-contract=fast"
! CHECK: "-menable-no-nans"
! CHECK: "-fapprox-func"
! CHECK: "-fno-signed-zeros"
! CHECK: "-mreassociate"
! CHECK: "-freciprocal-math"
! CHECK: "-fconvert=little-endian"
! CHECK: "-fpass-plugin=Bye
! CHECK: "-fversion-loops-for-stride"
! CHECK: "-flang-experimental-hlfir"
! CHECK: "-flang-deprecated-no-hlfir"
! CHECK: "-fno-ppc-native-vector-element-order"
! CHECK: "-fppc-native-vector-element-order"
! CHECK: "-Rpass"
! CHECK: "-Rpass-missed"
! CHECK: "-Rpass-analysis"
! CHECK: "-Rno-pass"
! CHECK: "-Reverything"
! CHECK: "-Rno-everything"
! CHECK: "-Rpass=inline"
! CHECK: "-mframe-pointer=none"
! CHECK: "-mllvm" "-print-before-all"
! CHECK: "-save-temps=obj"
