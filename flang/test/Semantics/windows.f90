! RUN: %if x86_64-registered-target %{ %python %S/test_errors.py %s %flang --target=x86_64-pc-windows-msvc -Werror %}
! RUN: %if aarch64-registered-target %{ %python %S/test_errors.py %s %flang --target=aarch64-pc-windows-msvc -Werror %}

subroutine uid
  !WARNING: User IDs do not exist on Windows. This function will always return 1
  i = getuid()
end subroutine uid

subroutine gid
  !WARNING: Group IDs do not exist on Windows. This function will always return 1
  i = getgid()
end subroutine gid
