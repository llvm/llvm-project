! RUN: %flang -O0 -S -emit-llvm %s -o - | FileCheck %s

! CHECK: %struct[[BLOCK1:\.BSS[0-9]+]] = type <{ [264 x i8] }>
! CHECK: %struct[[BLOCK2:\.BSS[0-9]+]] = type <{ [520 x i8] }>
! CHECK: %struct[[BLOCK3:\.BSS[0-9]+]] = type <{ [1032 x i8] }>
! CHECK: %struct[[BLOCK4:\.BSS[0-9]+]] = type <{ [2056 x i8] }>
! CHECK: %struct[[BLOCK5:_module_align_derived_[0-9]+_]] = type <{ [136 x i8] }>
! CHECK: @[[BLOCK1]] = internal global %struct[[BLOCK1]] zeroinitializer, align 256
! CHECK: @[[BLOCK2]] = internal global %struct[[BLOCK2]] zeroinitializer, align 512
! CHECK: @[[BLOCK3]] = internal global %struct[[BLOCK3]] zeroinitializer, align 1024
! CHECK: @[[BLOCK4]] = internal global %struct[[BLOCK4]] zeroinitializer, align 2048
! CHECK: @[[BLOCK5]] = common global %struct[[BLOCK5]] zeroinitializer, align 128

module module_align_derived
implicit none

    type T1
        integer(kind=2)     :: f1
        integer(kind=4)     :: f2
    end type T1

    !DIR$ ALIGN 128
    type(T1)    :: v1, v2

    interface
        module subroutine module_interface_subroutine()
        end subroutine module_interface_subroutine
    end interface

end module module_align_derived

submodule (module_align_derived) submodule_align_derived

    contains
    module subroutine module_interface_subroutine()

        type T3
            integer(kind=2)     :: f1
            integer(kind=4)     :: f2
        end type T3

        !DIR$ ALIGN 256
        type(T3)    :: v3, v4

        v3%f1 = 101
! CHECK:      store i16 101, ptr @[[BLOCK1]], align

        v3%f2 = 102
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK1]], i64 4
! CHECK:      store i32 102, ptr %[[TEMP]], align

        v4%f1 = 103
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK1]], i64 256
! CHECK:      store i16 103, ptr %[[TEMP]], align

        v4%f2 = 104
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK1]], i64 260
! CHECK:      store i32 104, ptr %[[TEMP]], align

    end subroutine module_interface_subroutine
end submodule submodule_align_derived



program align
use module_align_derived
implicit none

    type T5
        integer(kind=2)     :: f1
        integer(kind=4)     :: f2
    end type T5

    !DIR$ ALIGN 512
    type(T5)    :: v5, v6

    v5%f1 = 201
! CHECK:      store i16 201, ptr @[[BLOCK2]], align

    v5%f2 = 202
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK2]], i64 4
! CHECK:      store i32 202, ptr %[[TEMP]], align

    v6%f1 = 203
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK2]], i64 512
! CHECK:      store i16 203, ptr %[[TEMP]], align

    v6%f2 = 204
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK2]], i64 516
! CHECK:      store i32 204, ptr %[[TEMP]], align

    v1%f1 = 81
! CHECK:      store i16 81, ptr @[[BLOCK5]], align

    v1%f2 = 82
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK5]], i64 4
! CHECK:      store i32 82, ptr %[[TEMP]], align

    v2%f1 = 83
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK5]], i64 128
! CHECK:      store i16 83, ptr %[[TEMP]], align

    v2%f2 = 84
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK5]], i64 132
! CHECK:      store i32 84, ptr %[[TEMP]], align

end program align


subroutine subroutine_align()

    type T7
        integer(kind=2)     :: f1
        integer(kind=4)     :: f2
    end type T7

    !DIR$ ALIGN 1024
    type(T7)    :: v7, v8

    v7%f1 = 401
! CHECK:      store i16 401, ptr @[[BLOCK3]], align

    v7%f2 = 402
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK3]], i64 4
! CHECK:      store i32 402, ptr %[[TEMP]], align

    v8%f1 = 403
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK3]], i64 1024
! CHECK:      store i16 403, ptr %[[TEMP]], align

    v8%f2 = 404
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK3]], i64 1028
! CHECK:      store i32 404, ptr %[[TEMP]], align

    return
end subroutine subroutine_align


function function_align()

    type T9
        integer(kind=2)     :: f1
        integer(kind=4)     :: f2
    end type T9

    !DIR$ ALIGN 2048
    type(T9)    :: v9, v10

    v9%f1 = 801
! CHECK:      store i16 801, ptr @[[BLOCK4]], align

    v9%f2 = 802
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK4]], i64 4
! CHECK:      store i32 802, ptr %[[TEMP]], align

    v10%f1 = 803
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK4]], i64 2048
! CHECK:      store i16 803, ptr %[[TEMP]], align

    v10%f2 = 804
! CHECK:      %[[TEMP:[0-9]+]] = getelementptr i8, ptr @[[BLOCK4]], i64 2052
! CHECK:      store i32 804, ptr %[[TEMP]], align

    return
end function function_align
