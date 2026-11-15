! Check the comment sentinel of a plugin loaded with `flang -fc1 -load` in
! fixed form, and that without the plugin its lines are comments.

! REQUIRES: plugins, examples
! XFAIL: system-aix

! RUN: %flang_fc1 -load %llvmshlibdir/flangDirectivesPlugin%pluginext \
! RUN:   -emit-hlfir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -o - %s 2>&1 | FileCheck %s --check-prefix=NOPLUGIN
! RUN: %flang_fc1 -fopenmp -emit-hlfir -o - %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NOPLUGIN

      subroutine s
c$example note(text='c in column 1')
      end

      subroutine t
      external s
*$example callback(handler=s,
*$example+ tag='continued')
      end

! CHECK-DAG: func.func @_QPs() attributes {fir.directives = [{args = {text = "c in column 1"}, keyword = "note", prefix = "example"}]}
! CHECK-DAG: func.func @_QPt() attributes {fir.directives = [{args = {handler = @_QPs, tag = "continued"}, keyword = "callback", prefix = "example"}]}

! NOPLUGIN-NOT: warning
! NOPLUGIN-NOT: error
! NOPLUGIN-NOT: fir.directives
