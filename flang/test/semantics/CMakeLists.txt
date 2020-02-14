#===-- test/semantics/CMakeLists.txt ---------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# Run tests with test_errors.sh. It compiles the test with f18 and compares
# actual errors produced with expected ones listed in the source.

# These test files have expected errors in the source
set(ERROR_TESTS
  implicit01.f90
  implicit02.f90
  implicit03.f90
  implicit04.f90
  implicit05.f90
  implicit06.f90
  implicit07.f90
  implicit08.f90
  int-literals.f90
  io01.f90
  io02.f90
  io03.f90
  io04.f90
  io05.f90
  io06.f90
  io07.f90
  io08.f90
  io09.f90
  io10.f90
  kinds02.f90
  resolve01.f90
  resolve02.f90
  resolve03.f90
  resolve04.f90
  resolve05.f90
  resolve06.f90
  resolve07.f90
  resolve08.f90
  resolve09.f90
  resolve10.f90
  resolve11.f90
  resolve12.f90
  resolve13.f90
  resolve14.f90
  resolve15.f90
  resolve16.f90
  resolve17.f90
  resolve18.f90
  resolve19.f90
  resolve20.f90
  resolve21.f90
  resolve22.f90
  resolve23.f90
  resolve24.f90
  resolve25.f90
  resolve26.f90
  resolve27.f90
  resolve28.f90
  resolve29.f90
  resolve30.f90
  resolve31.f90
  resolve32.f90
  resolve33.f90
  resolve34.f90
  resolve35.f90
  resolve36.f90
  resolve37.f90
  resolve38.f90
  resolve39.f90
  resolve40.f90
  resolve41.f90
  resolve42.f90
  resolve43.f90
  resolve44.f90
  resolve45.f90
  resolve46.f90
  resolve47.f90
  resolve48.f90
  resolve49.f90
  resolve50.f90
  resolve51.f90
  resolve52.f90
  resolve53.f90
  resolve54.f90
  resolve55.f90
  resolve56.f90
  resolve57.f90
  resolve58.f90
  resolve59.f90
  resolve60.f90
  resolve61.f90
  resolve62.f90
  resolve63.f90
  resolve64.f90
  resolve65.f90
  resolve66.f90
  resolve67.f90
  resolve68.f90
  resolve69.f90
  resolve70.f90
  resolve71.f90
  resolve72.f90
  stop01.f90
  structconst01.f90
  structconst02.f90
  structconst03.f90
  structconst04.f90
  assign01.f90
  assign02.f90
  assign03.f90
  if_arith02.f90
  if_arith03.f90
  if_arith04.f90
  if_construct02.f90
  if_stmt02.f90
  if_stmt03.f90
  computed-goto01.f90
  computed-goto02.f90
  nullify01.f90
  nullify02.f90
  deallocate01.f90
  deallocate04.f90
  deallocate05.f90
  coarrays01.f90
  altreturn01.f90
# Issue 407
#  altreturn02.f90
#  altreturn03.f90
  altreturn04.f90
  altreturn05.f90
  allocate01.f90
  allocate02.f90
  allocate03.f90
  allocate04.f90
  allocate05.f90
  allocate06.f90
  allocate07.f90
  allocate08.f90
  allocate09.f90
  allocate10.f90
  allocate11.f90
  allocate12.f90
  allocate13.f90
  doconcurrent01.f90
  doconcurrent05.f90
  doconcurrent06.f90
  doconcurrent08.f90
  dosemantics01.f90
  dosemantics02.f90
  dosemantics03.f90
  dosemantics04.f90
  dosemantics05.f90
  dosemantics06.f90
  dosemantics07.f90
  dosemantics08.f90
  dosemantics09.f90
  dosemantics10.f90
  dosemantics11.f90
  dosemantics12.f90
  expr-errors01.f90
  expr-errors02.f90
  null01.f90
  omp-resolve01.f90
  omp-resolve02.f90
  omp-resolve03.f90
  omp-resolve04.f90
  omp-resolve05.f90
  omp-clause-validity01.f90
  omp-loop-association.f90
#  omp-nested01.f90
  omp-declarative-directive.f90
  omp-atomic.f90
  omp-device-constructs.f90
  equivalence01.f90
  init01.f90
  if_arith01.f90
  if_construct01.f90
  if_stmt01.f90
  blockconstruct01.f90
  blockconstruct02.f90
  blockconstruct03.f90
  call01.f90
  call02.f90
  call03.f90
  call04.f90
  call05.f90
  call06.f90
  call07.f90
  call08.f90
  call09.f90
  call10.f90
  call11.f90
  call12.f90
  call13.f90
  call14.f90
  call15.f90
  forall01.f90
  misc-declarations.f90
  separate-module-procs.f90
  bindings01.f90
  bad-forward-type.f90
  c_f_pointer.f90
  critical01.f90
  critical02.f90
  critical03.f90
  block-data01.f90
)

# These test files have expected symbols in the source
set(SYMBOL_TESTS
  symbol01.f90
  symbol02.f90
  symbol03.f90
  symbol05.f90
  symbol06.f90
  symbol07.f90
  symbol08.f90
  symbol09.f90
  symbol10.f90
  symbol11.f90
  symbol12.f90
  symbol13.f90
  symbol14.f90
  symbol15.f90
  symbol16.f90
  symbol17.f90
  omp-symbol01.f90
  omp-symbol02.f90
  omp-symbol03.f90
  omp-symbol04.f90
  omp-symbol05.f90
  omp-symbol06.f90
  omp-symbol07.f90
  omp-symbol08.f90
  kinds01.f90
  kinds03.f90
  procinterface01.f90
)

# These test files have expected .mod file contents in the source
set(MODFILE_TESTS
  modfile01.f90
  modfile02.f90
  modfile03.f90
  modfile04.f90
  modfile05.f90
  modfile06.f90
  modfile07.f90
  modfile08.f90
  modfile09-*.f90
  modfile10.f90
  modfile11.f90
  modfile12.f90
  modfile13.f90
  modfile14.f90
  modfile15.f90
  modfile16.f90
  modfile17.f90
  modfile18.f90
  modfile19.f90
  modfile20.f90
  modfile21.f90
  modfile22.f90
  modfile23.f90
  modfile24.f90
  modfile25.f90
  modfile26.f90
  modfile27.f90
  modfile28.f90
  modfile29.f90
  modfile30.f90
  modfile31.f90
  modfile32.f90
  modfile33.f90
  modfile34.f90
  modfile35.f90
)

set(LABEL_TESTS
  label*.[Ff]90
)

set(DOCONCURRENT_TESTS
  doconcurrent02.f90
  doconcurrent03.f90
  doconcurrent04.f90
  doconcurrent07.f90
)

set(CANONDO_TESTS
  canondo*.[Ff]90
)

set(CRITICAL_TESTS
  critical04.f90
)

set(GETSYMBOLS_TESTS
  getsymbols01.f90
  getsymbols02-*.f90
  getsymbols03-a.f90
  getsymbols04.f90
  getsymbols05.f90
)

set(GETDEFINITION_TESTS
  getdefinition01.f90
  getdefinition02.f
  getdefinition03-a.f90
  getdefinition04.f90
  getdefinition05.f90
)

set(F18 $<TARGET_FILE:f18>)

foreach(test ${ERROR_TESTS})
  add_test(NAME ${test}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_errors.sh ${test} ${F18})
endforeach()

foreach(test ${SYMBOL_TESTS})
  add_test(NAME ${test}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_symbols.sh ${test} ${F18})
endforeach()

foreach(test ${MODFILE_TESTS})
  add_test(NAME ${test}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_modfile.sh ${test} ${F18})
endforeach()

foreach(test ${LABEL_TESTS} ${CANONDO_TESTS} ${DOCONCURRENT_TESTS}
             ${CRITICAL_TESTS} ${GETSYMBOLS_TESTS} ${GETDEFINITION_TESTS})
  add_test(NAME ${test}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_any.sh ${test} ${F18})
endforeach()
