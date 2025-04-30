! RUN: %flang -O0 -S -emit-llvm %s -o - | FileCheck %s

! CHECK: %struct[[BLOCK1:\.STATICS[0-9]+]] = type <{ [10 x i8] }
! CHECK: %struct[[BLOCK2:\.STATICS[0-9]+]] = type <{ [10 x i8] }
! CHECK: %struct[[BLOCK3:\.STATICS[0-9]+]] = type <{ [10 x i8] }
! CHECK: %struct[[BLOCK4:\.STATICS[0-9]+]] = type <{ [10 x i8] }
! CHECK: %struct[[BLOCK5:_module_align_character_[0-9]+_]] = type <{ [10 x i8] }
! CHECK: %struct[[BLOCK6:_module_align_character_[0-9]+_]] = type <{ [10 x i8] }
! CHECK: @[[BLOCK1]] = internal global %struct[[BLOCK1]] <{{[^>]+}}>, align 256
! CHECK: @[[BLOCK2]] = internal global %struct[[BLOCK2]] <{{[^>]+}}>, align 512
! CHECK: @[[BLOCK3]] = internal global %struct[[BLOCK3]] <{{[^>]+}}>, align 1024
! CHECK: @[[BLOCK4]] = internal global %struct[[BLOCK4]] <{{[^>]+}}>, align 2048
! CHECK: @[[BLOCK5]] = common global %struct[[BLOCK5]] zeroinitializer, align 128
! CHECK: @[[BLOCK6]] = global %struct[[BLOCK6]] <{{[^>]+}}>, align 128

module module_align_character
implicit none

    !DIR$ ALIGN 128
    character(len=10) :: v1, v2 = "128"

    interface
        module subroutine module_interface_subroutine()
        end subroutine module_interface_subroutine
    end interface

end module module_align_character

submodule (module_align_character) submodule_align_character

    contains
    module subroutine module_interface_subroutine()

        !DIR$ ALIGN 256
        character(len=10) :: v3, v4 = "256"
! CHECK:    %[[V3:v3_[0-9]+]] = alloca [10 x i8], align 256

        v3 = "101"
! CHECK:      store volatile i64 %{{[0-9]+}}, ptr %[[V3]], align

        v4 = "102"
! CHECK:      store volatile i64 %{{[0-9]+}}, ptr @[[BLOCK1]], align

    end subroutine module_interface_subroutine
end submodule submodule_align_character

program align
use module_align_character
implicit none

    !DIR$ ALIGN 512
    character(len=10) :: v5, v6 = "512"
! CHECK:    %[[V5:v5_[0-9]+]] = alloca [10 x i8], align 512

    v5 = "201"
! CHECK:      store volatile i64 %{{[0-9]+}}, ptr %[[V5]], align

    v6 = "202"
! CHECK:      store volatile i64 %{{[0-9]+}}, ptr @[[BLOCK2]], align

    v1 = "81"
! CHECK:      store volatile i64 %{{[0-9]+}}, ptr @[[BLOCK5]], align

    v2 = "82"
! CHECK:      store volatile i64 %{{[0-9]+}}, ptr @[[BLOCK6]], align

end program align

subroutine subroutine_align()

    !DIR$ ALIGN 1024
    character(len=10) :: v7, v8 = "1024"
! CHECK:    %[[V7:v7_[0-9]+]] = alloca [10 x i8], align 1024

    v7 = "401"
! CHECK:      store volatile i64 %{{[0-9]+}}, ptr %[[V7]], align

    v8 = "402"
! CHECK:      store volatile i64 %{{[0-9]+}}, ptr @[[BLOCK3]], align

    return
end subroutine subroutine_align

function function_align()

    !DIR$ ALIGN 2048
    character(len=10) :: v9, v10 = "2048"
! CHECK:    %[[V9:v9_[0-9]+]] = alloca [10 x i8], align 2048

    v9 = "801"
! CHECK:      store volatile i64 %{{[0-9]+}}, ptr %[[V9]], align

    v10 = "802"
! CHECK:      store volatile i64 %{{[0-9]+}}, ptr @[[BLOCK4]], align

    return
end function function_align
