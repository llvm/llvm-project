! Check that a directive with the prefix of a plugin loaded with
! `flang -fc1 -load` but malformed arguments is an error, not an ignored
! directive.

! REQUIRES: plugins, examples
! XFAIL: system-aix

! RUN: not %flang_fc1 -load %llvmshlibdir/flangDirectivesPlugin%pluginext \
! RUN:   -fsyntax-only %s 2>&1 | FileCheck %s

subroutine s
  ! CHECK: error: malformed argument list of a directive defined by a plugin
  !dir$ example callback(handler=s, priority=1.5)
  ! CHECK: error: malformed argument list of a directive defined by a plugin
  !dir$ example callback(handler=s, priority=-1)
  ! CHECK: error: malformed argument list of a directive defined by a plugin
  !dir$ example callback(handler=s) trailing
  ! CHECK-NOT: Unrecognized compiler directive
end
