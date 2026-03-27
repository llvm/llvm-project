! RUN: %python %S/test_errors.py %s %flang_fc1
type t
end type
real, pointer :: rptr
type(t), pointer :: tptr
class(*), pointer :: ulpp
print *, associated(rptr, ulpp)
print *, associated(ulpp, rptr)
print *, associated(tptr, ulpp)
print *, associated(ulpp, tptr)
!ERROR: Arguments of ASSOCIATED() must be a pointer and an optional valid target
print *, associated(rptr, tptr)
!ERROR: Arguments of ASSOCIATED() must be a pointer and an optional valid target
print *, associated(tptr, rptr)
end
