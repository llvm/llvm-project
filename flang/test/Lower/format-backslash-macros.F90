! RUN: %flang_fc1 -cpp -emit-fir %s -o - | FileCheck %s --check-prefix=DEFAULT
! RUN: %flang_fc1 -cpp -fbackslash -emit-fir %s -o - | FileCheck %s --check-prefix=ESCAPES

#define PREFIX 12h
#define FORMAT_STRING (1x, 'macro\path')
program macro_hollerith
  write (*, 10)
  write (*, 20)
10 format (PREFIX'abxxxxxxxxx, 'zz,3h\\nb,1h')
! FIXME: The preprocessor replaces the provenance of the entire expansion,
! including inserted backslashes. Keep the existing default-mode behavior
! until it can preserve the provenance needed to undo the doubling here.
20 format FORMAT_STRING
end

! DEFAULT-DAG: fir.string_lit "(12h'abxxxxxxxxx, 'zz,3h\\\\nb,1h')"(33)
! ESCAPES-DAG: fir.string_lit "(12h'abxxxxxxxxx, 'zz,3h\\nb,1h')"(32)
! DEFAULT-DAG: fir.string_lit "(1x, 'macro\\\\path')"(19)
! ESCAPES-DAG: fir.string_lit "(1x, 'macropath')"(17)
