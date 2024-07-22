!RUN: %flang -fsyntax-only -pedantic -I %S/Inputs/ %s 2>&1 | FileCheck %s
!CHECK: warning: SAVE attribute was already specified on 'x'
!CHECK: portability: #include: extra stuff ignored after file name
save x
save x
#include <empty.h>    crud after header name
end
