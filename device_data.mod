!mod$ v1 sum:4742aa75f60f1f4e
!need$ c8dda17ea6314235 i __cuda_builtins
!need$ 3537609d44c937af i cudadevice
module device_data
real(4),constant::d1(1_8:10_8)
real(4),constant::d2(1_8:10_8,1_8:10_8)
real(4),constant::d3(1_8:10_8)
real(4),constant::d4(1_8:10_8)
real(4),constant::d5
real(4),constant::d11(1_8:10_8)
real(4),constant::d12(1_8:10_8,1_8:10_8)
real(4),constant::d13(1_8:10_8)
real(4),constant::d14(1_8:10_8)
real(4),constant::d15(1_8:10_8)
real(4),constant::d16
real(4),constant::d21(1_8:10_8)
real(4),constant::d22(1_8:10_8,1_8:10_8)
real(4),constant::d23(1_8:10_8)
real(4),constant::d24(1_8:10_8)
real(4),constant::d25
contains
attributes(global) subroutine init()
end
end
