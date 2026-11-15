! Check that the directives of a plugin loaded with `flang -fc1 -load` are
! attached to the operations of their subjects.

! REQUIRES: plugins, examples
! XFAIL: system-aix

! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 -load %llvmshlibdir/flangDirectivesPlugin%pluginext \
! RUN:   -emit-hlfir -module-dir %t -o - %s | FileCheck %s

module m
  real :: x, y
  common /blk/ z
  real :: z
  interface gen
    module procedure gen_i, gen_r
  end interface

  ! A directive in a module names its subject.
  !dir$ example watch(x, by=y)
  !dir$ example watch(/blk/)
  !dir$ example note(y, text="a variable")
  ! A generic interface stands for each of its specific procedures.
  !dir$ example note(gen, text="generic")
contains
  subroutine gen_i(i)
    integer :: i
  end
  subroutine gen_r(r)
    real :: r
  end

  subroutine on_call
  end

  ! In a subprogram, the subject is the subprogram unless named.
  subroutine s
    !dir$ example callback(handler=on_call, priority=2, tag="s")
    !dir$ example note(text=no_quotes)
  end

  ! A function's own name, without a RESULT clause, is the function.
  function f()
    real :: f
    !DIR$ EXAMPLE CALLBACK(f, HANDLER=on_call)
    f = 0.
  end

  ! The plugin's sentinel, with a continuation line.
  subroutine sentinel
    !$example callback(handler=on_call, &
    !$example tag="sentinel")
  end
end

! A procedure the unit does not define or call.
subroutine t
  interface
    subroutine ext
    end
  end interface
  !dir$ example callback(ext, handler=t)
end

! CHECK-DAG: fir.global @_QMmEx {fir.directives = [{args = {by = @_QMmEy}, keyword = "watch", prefix = "example"}]}
! CHECK-DAG: fir.global @_QMmEy {fir.directives = [{args = {text = "a variable"}, keyword = "note", prefix = "example"}]}
! CHECK-DAG: fir.global common @blk_({{.*}}) <{{.*}}> {fir.directives = [{args = {}, keyword = "watch", prefix = "example"}]}
! CHECK-DAG: func.func @_QMmPs() attributes {fir.directives = [{args = {handler = @_QMmPon_call, priority = 2 : i64, tag = "s"}, keyword = "callback", prefix = "example"}, {args = {text = "no_quotes"}, keyword = "note", prefix = "example"}]}
! CHECK-DAG: func.func @_QMmPgen_i({{.*}}) attributes {fir.directives = [{args = {text = "generic"}, keyword = "note", prefix = "example"}]}
! CHECK-DAG: func.func @_QMmPgen_r({{.*}}) attributes {fir.directives = [{args = {text = "generic"}, keyword = "note", prefix = "example"}]}
! CHECK-DAG: func.func @_QMmPf() -> f32 attributes {fir.directives = [{args = {handler = @_QMmPon_call}, keyword = "callback", prefix = "example"}]}
! CHECK-DAG: func.func @_QMmPsentinel() attributes {fir.directives = [{args = {handler = @_QMmPon_call, tag = "sentinel"}, keyword = "callback", prefix = "example"}]}
! CHECK-DAG: func.func private @_QPext() attributes {fir.directives = [{args = {handler = @_QPt}, keyword = "callback", prefix = "example"}]}
