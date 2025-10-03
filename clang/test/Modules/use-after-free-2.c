// RUN: rm -rf %t
// RUN: split-file %s %t

//--- A.modulemap
module A {
  header "A.h"

  textual header "A00.h"
  textual header "A01.h"
  textual header "A02.h"
  textual header "A03.h"
  textual header "A04.h"
  textual header "A05.h"
  textual header "A06.h"
  textual header "A07.h"
  textual header "A08.h"
  textual header "A09.h"

  textual header "A10.h"
  textual header "A11.h"
  textual header "A12.h"
  textual header "A13.h"
  textual header "A14.h"
  textual header "A15.h"
  textual header "A16.h"
  textual header "A17.h"
  textual header "A18.h"
  textual header "A19.h"

  textual header "A20.h"
  textual header "A21.h"
  textual header "A22.h"
  textual header "A23.h"
  textual header "A24.h"
  textual header "A25.h"
  textual header "A26.h"
  textual header "A27.h"
  textual header "A28.h"
  textual header "A29.h"

  textual header "A30.h"
  textual header "A31.h"
  textual header "A32.h"
  textual header "A33.h"
  textual header "A34.h"
  textual header "A35.h"
  textual header "A36.h"
  textual header "A37.h"
  textual header "A38.h"
  textual header "A39.h"

  textual header "A40.h"
  textual header "A41.h"
  textual header "A42.h"
  textual header "A43.h"
  textual header "A44.h"
  textual header "A45.h"
}
//--- A.h

//--- A00.h
//--- A01.h
//--- A02.h
//--- A03.h
//--- A04.h
//--- A05.h
//--- A06.h
//--- A07.h
//--- A08.h
//--- A09.h

//--- A10.h
//--- A11.h
//--- A12.h
//--- A13.h
//--- A14.h
//--- A15.h
//--- A16.h
//--- A17.h
//--- A18.h
//--- A19.h

//--- A20.h
//--- A21.h
//--- A22.h
//--- A23.h
//--- A24.h
//--- A25.h
//--- A26.h
//--- A27.h
//--- A28.h
//--- A29.h

//--- A30.h
//--- A31.h
//--- A32.h
//--- A33.h
//--- A34.h
//--- A35.h
//--- A36.h
//--- A37.h
//--- A38.h
//--- A39.h

//--- A40.h
//--- A41.h
//--- A42.h
//--- A43.h
//--- A44.h
//--- A45.h

//--- B.modulemap
module B { header "B.h" }
//--- B.h
#include "A.h"

//--- C.modulemap
module C { header "C.h" }
//--- C.h
#include "A00.h"
#include "A01.h"
#include "A02.h"
#include "A03.h"
#include "A04.h"
#include "A05.h"
#include "A06.h"
#include "A07.h"
#include "A08.h"
#include "A09.h"

#include "A10.h"
#include "A11.h"
#include "A12.h"
#include "A13.h"
#include "A14.h"
#include "A15.h"
#include "A16.h"
#include "A17.h"
#include "A18.h"
#include "A19.h"

#include "A20.h"
#include "A21.h"
#include "A22.h"
#include "A23.h"
#include "A24.h"
#include "A25.h"
#include "A26.h"
#include "A27.h"
#include "A28.h"
#include "A29.h"

#include "A30.h"
#include "A31.h"
#include "A32.h"
#include "A33.h"
#include "A34.h"
#include "A35.h"
#include "A36.h"
#include "A37.h"
#include "A38.h"
#include "A39.h"

#include "A40.h"
#include "A41.h"
#include "A42.h"
#include "A43.h"
#include "A44.h"
#include "A45.h"

#include "B.h"

// RUN: %clang_cc1 -fmodules -fno-modules-prune-non-affecting-module-map-files \
// RUN:   -emit-module %t/A.modulemap -fmodule-name=A -o %t/A.pcm
// RUN: %clang_cc1 -fmodules -fno-modules-prune-non-affecting-module-map-files \
// RUN:   -emit-module %t/B.modulemap -fmodule-name=B -o %t/B.pcm \
// RUN:   -fmodule-file=A=%t/A.pcm -fmodule-map-file=%t/A.modulemap
// RUN: %clang_cc1 -fmodules -fno-modules-prune-non-affecting-module-map-files \
// RUN:   -emit-module %t/C.modulemap -fmodule-name=C -o %t/C.pcm \
// RUN:   -fmodule-file=B=%t/B.pcm -fmodule-map-file=%t/B.modulemap
