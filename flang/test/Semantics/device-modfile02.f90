! The device side must not write module files (they belong to the host).
! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 -fsyntax-only -foffload-device -I%S/Inputs -J%t %s
! RUN: not ls %t/device_modfile02.mod

module device_modfile02
end module
