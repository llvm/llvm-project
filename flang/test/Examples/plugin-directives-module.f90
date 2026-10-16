! Check that the directives of a plugin loaded with `flang -fc1 -load` on the
! entities of a module are written to its module file, and seen by the units
! that use it.

! REQUIRES: plugins, examples
! XFAIL: system-aix

! RUN: rm -rf %t && split-file %s %t
! RUN: %flang_fc1 -load %llvmshlibdir/flangDirectivesPlugin%pluginext \
! RUN:   -fsyntax-only -module-dir %t %t/m.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=WARN
! RUN: FileCheck %s --check-prefix=MOD < %t/m.mod
! RUN: %flang_fc1 -load %llvmshlibdir/flangDirectivesPlugin%pluginext \
! RUN:   -emit-hlfir -module-dir %t -o - %t/use.f90 | FileCheck %s

! Without the plugin, the directives of the module file are ignored silently.
! RUN: %flang_fc1 -emit-hlfir -module-dir %t -o - %t/use.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NOPLUGIN

!--- m.f90
module m
  real :: x
  !dir$ example watch(x)
contains
  subroutine on_call
  end
  subroutine s(y)
    real :: y
    !dir$ example callback(handler=on_call, tag="s")
    ! A procedure naming itself.
    !dir$ example note(s, text="self")
    ! Not visible in the module: left out of the module file.
    !dir$ example callback(handler=internal)
  contains
    subroutine internal
    end
  end
end

!--- use.f90
subroutine u
  use m
  call s(x)
end

! WARN: warning: This 'example callback' directive is not written to the module file of 'm', where 'internal' is not visible; units that use the module do not see it

! MOD:      !dir$ example watch(x)
! MOD-NEXT: !dir$ example callback(s, handler=on_call, tag="s")
! MOD-NEXT: !dir$ example note(s, text="self")
! MOD-NOT:  internal

! CHECK-DAG: fir.global @_QMmEx {fir.directives = [{args = {}, keyword = "watch", prefix = "example"}]}
! CHECK-DAG: func.func private @_QMmPs(!fir.ref<f32>) attributes {fir.directives = [{args = {handler = @_QMmPon_call, tag = "s"}, keyword = "callback", prefix = "example"}, {args = {text = "self"}, keyword = "note", prefix = "example"}]}

! NOPLUGIN-NOT: warning
! NOPLUGIN-NOT: fir.directives
