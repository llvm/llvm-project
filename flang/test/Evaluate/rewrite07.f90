! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

subroutine test_pack_size_rewrite(x, mask)
    real :: x(:)
    logical, intent(in) :: mask(:)
    ! CHECK: CALL test(count(mask,kind=8_8))
    call test(size(pack(x, mask), dim=1, kind=8))
end subroutine
