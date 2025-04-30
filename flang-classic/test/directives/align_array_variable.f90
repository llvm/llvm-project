! RUN: %flang -O0 -S -emit-llvm %s -o - | FileCheck %s

! CHECK: %struct[[BLOCK1:\.BSS[0-9]+]] = type <{ [356 x i8] }>
! CHECK: %struct[[BLOCK2:\.BSS[0-9]+]] = type <{ [612 x i8] }>
! CHECK: %struct[[BLOCK3:\.BSS[0-9]+]] = type <{ [1124 x i8] }>
! CHECK: %struct[[BLOCK4:\.BSS[0-9]+]] = type <{ [2148 x i8] }>
! CHECK: %struct[[BLOCK5:_module_align_array_[0-9]+_]] = type <{ [228 x i8] }>
! CHECK: @[[BLOCK1]] = internal global %struct[[BLOCK1]] zeroinitializer, align 256
! CHECK: @[[BLOCK2]] = internal global %struct[[BLOCK2]] zeroinitializer, align 512
! CHECK: @[[BLOCK3]] = internal global %struct[[BLOCK3]] zeroinitializer, align 1024
! CHECK: @[[BLOCK4]] = internal global %struct[[BLOCK4]] zeroinitializer, align 2048
! CHECK: @[[BLOCK5]] = common global %struct[[BLOCK5]] zeroinitializer, align 128

module module_align_array
implicit none

    !DIR$ ALIGN 128
    integer, dimension (5,5) :: v1, v2

    interface
        module subroutine module_interface_subroutine()
        end subroutine module_interface_subroutine
    end interface

end module module_align_array

submodule (module_align_array) submodule_align_array

    contains
    module subroutine module_interface_subroutine()

        !DIR$ ALIGN 256
        integer, dimension (5,5) :: v3, v4

        v3(1, 1) = 101
! CHECK:      store i32 101, ptr @[[BLOCK1]], align

        v3(5, 5) = 102
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK1]], i64 96
! CHECK:      store i32 102, ptr %[[TEMP]], align

        v4(1, 1) = 103
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK1]], i64 256
! CHECK:      store i32 103, ptr %[[TEMP]], align

        v4(5, 5) = 104
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK1]], i64 352
! CHECK:      store i32 104, ptr %[[TEMP]], align

    end subroutine module_interface_subroutine
end submodule submodule_align_array



program align
use module_align_array
implicit none

    !DIR$ ALIGN 512
    integer, dimension (5,5) :: v5, v6

    v5(1, 1) = 201
! CHECK:      store i32 201, ptr @[[BLOCK2]], align

    v5(5, 5) = 202
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK2]], i64 96
! CHECK:      store i32 202, ptr %[[TEMP]], align

    v6(1, 1) = 203
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK2]], i64 512
! CHECK:      store i32 203, ptr %[[TEMP]], align

    v6(5, 5) = 204
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK2]], i64 608
! CHECK:      store i32 204, ptr %[[TEMP]], align

    v1(1, 1) = 81
! CHECK:      store i32 81, ptr @[[BLOCK5]], align

    v1(5, 5) = 82
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK5]], i64 96
! CHECK:      store i32 82, ptr %[[TEMP]], align

    v2(1, 1) = 83
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK5]], i64 128
! CHECK:      store i32 83, ptr %[[TEMP]], align

    v2(5, 5) = 84
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK5]], i64 224
! CHECK:      store i32 84, ptr %[[TEMP]], align

end program align


subroutine subroutine_align()

    !DIR$ ALIGN 1024
    integer, dimension (5,5) :: v7, v8

    v7(1, 1) = 401
! CHECK:      store i32 401, ptr @[[BLOCK3]], align

    v7(5, 5) = 402
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK3]], i64 96
! CHECK:      store i32 402, ptr %[[TEMP]], align

    v8(1, 1) = 403
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK3]], i64 1024
! CHECK:      store i32 403, ptr %[[TEMP]], align

    v8(5, 5) = 404
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK3]], i64 1120
! CHECK:      store i32 404, ptr %[[TEMP]], align

    return
end subroutine subroutine_align


function function_align()

    !DIR$ ALIGN 2048
    integer, dimension (5,5) :: v9, v10

    v9(1, 1) = 801
! CHECK:      store i32 801, ptr @[[BLOCK4]], align

    v9(5, 5) = 802
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK4]], i64 96
! CHECK:      store i32 802, ptr %[[TEMP]], align

    v10(1, 1) = 803
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK4]], i64 2048
! CHECK:      store i32 803, ptr %[[TEMP]], align

    v10(5, 5) = 804
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK4]], i64 2144
! CHECK:      store i32 804, ptr %[[TEMP]], align

    return
end function function_align
