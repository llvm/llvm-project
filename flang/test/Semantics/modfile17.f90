! RUN: %python %S/test_modfile.py %s %flang_fc1
! Tests parameterized derived type instantiation with KIND parameters

module m
  type :: capture(k1,k2,k4,k8)
    integer(kind=1), kind :: k1
    integer(kind=2), kind :: k2
    integer(kind=4), kind :: k4
    integer(kind=8), kind :: k8
    integer(kind=k1) :: j1
    integer(kind=k2) :: j2
    integer(kind=k4) :: j4
    integer(kind=k8) :: j8
  end type capture
  type :: defaulted(n1,n2,n4,n8)
    integer(kind=1), kind :: n1 = 1
    integer(kind=2), kind :: n2 = n1 * 2
    integer(kind=4), kind :: n4 = 2 * n2
    integer(kind=8), kind :: n8 = 12 - n4
    type(capture(n1,n2,n4,n8)) :: cap
  end type defaulted
  type, extends(defaulted) :: extension(k5)
    integer(kind=4), kind :: k5 = 4
    integer(kind=k5) :: j5
  end type extension
  type(capture(1,1,1,1)) :: x1111
  integer(kind=x1111%j1%kind) :: res01_1
  integer(kind=x1111%j2%kind) :: res02_1
  integer(kind=x1111%j4%kind) :: res03_1
  integer(kind=x1111%j8%kind) :: res04_1
  type(capture(8,8,8,8)) :: x8888
  integer(kind=x8888%j1%kind) :: res05_8
  integer(kind=x8888%j2%kind) :: res06_8
  integer(kind=x8888%j4%kind) :: res07_8
  integer(kind=x8888%j8%kind) :: res08_8
  type(capture(2,k8=1,k4=8,k2=4)) :: x2481
  integer(kind=x2481%j1%kind) :: res09_2
  integer(kind=x2481%j2%kind) :: res10_4
  integer(kind=x2481%j4%kind) :: res11_8
  integer(kind=x2481%j8%kind) :: res12_1
  type(capture(2,1,k4=8,k8=4)) :: x2184
  integer(kind=x2184%j1%kind) :: res13_2
  integer(kind=x2184%j2%kind) :: res14_1
  integer(kind=x2184%j4%kind) :: res15_8
  integer(kind=x2184%j8%kind) :: res16_4
  type(defaulted) :: x1248
  integer(kind=x1248%cap%j1%kind) :: res17_1
  integer(kind=x1248%cap%j2%kind) :: res18_2
  integer(kind=x1248%cap%j4%kind) :: res19_4
  integer(kind=x1248%cap%j8%kind) :: res20_8
  type(defaulted(2)) :: x2484
  integer(kind=x2484%cap%j1%kind) :: res21_2
  integer(kind=x2484%cap%j2%kind) :: res22_4
  integer(kind=x2484%cap%j4%kind) :: res23_8
  integer(kind=x2484%cap%j8%kind) :: res24_4
  type(defaulted(n8=2)) :: x1242
  integer(kind=x1242%cap%j1%kind) :: res25_1
  integer(kind=x1242%cap%j2%kind) :: res26_2
  integer(kind=x1242%cap%j4%kind) :: res27_4
  integer(kind=x1242%cap%j8%kind) :: res28_2
  type(extension(1,1,1,1,1)) :: x11111
  integer(kind=x11111%defaulted%cap%j1%kind) :: res29_1
  integer(kind=x11111%cap%j2%kind) :: res30_1
  integer(kind=x11111%cap%j4%kind) :: res31_1
  integer(kind=x11111%cap%j8%kind) :: res32_1
  integer(kind=x11111%j5%kind) :: res33_1
  type(extension(2,8,4,1,8)) :: x28418
  integer(kind=x28418%defaulted%cap%j1%kind) :: res34_2
  integer(kind=x28418%cap%j2%kind) :: res35_8
  integer(kind=x28418%cap%j4%kind) :: res36_4
  integer(kind=x28418%cap%j8%kind) :: res37_1
  integer(kind=x28418%j5%kind) :: res38_8
  type(extension(8,n8=1,k5=2,n2=4,n4=8)) :: x84812
  integer(kind=x84812%defaulted%cap%j1%kind) :: res39_8
  integer(kind=x84812%cap%j2%kind) :: res40_4
  integer(kind=x84812%cap%j4%kind) :: res41_8
  integer(kind=x84812%cap%j8%kind) :: res42_1
  integer(kind=x84812%j5%kind) :: res43_2
  type(extension(k5=2)) :: x12482
  integer(kind=x12482%defaulted%cap%j1%kind) :: res44_1
  integer(kind=x12482%cap%j2%kind) :: res45_2
  integer(kind=x12482%cap%j4%kind) :: res46_4
  integer(kind=x12482%cap%j8%kind) :: res47_8
  integer(kind=x12482%j5%kind) :: res48_2
end module

!Expect: m.mod
!module m
!type::capture(k1,k2,k4,k8)
!integer(1),kind::k1
!integer(2),kind::k2
!integer(4),kind::k4
!integer(8),kind::k8
!integer(int(int(k1,kind=1),kind=8))::j1
!integer(int(int(k2,kind=2),kind=8))::j2
!integer(int(int(k4,kind=4),kind=8))::j4
!integer(k8)::j8
!end type
!type::defaulted(n1,n2,n4,n8)
!integer(1),kind::n1=1_4
!integer(2),kind::n2=2_4*int(int(n1,kind=1),kind=4)
!integer(4),kind::n4=2_4*int(int(n2,kind=2),kind=4)
!integer(8),kind::n8=12_4-int(n4,kind=4)
!type(capture(k1=int(n1,kind=1),k2=int(n2,kind=2),k4=int(n4,kind=4),k8=n8))::cap
!end type
!type,extends(defaulted)::extension(k5)
!integer(4),kind::k5=4_4
!integer(int(int(k5,kind=4),kind=8))::j5
!end type
!type(capture(k1=1_1,k2=1_2,k4=1_4,k8=1_8))::x1111
!integer(1)::res01_1
!integer(1)::res02_1
!integer(1)::res03_1
!integer(1)::res04_1
!type(capture(k1=8_1,k2=8_2,k4=8_4,k8=8_8))::x8888
!integer(8)::res05_8
!integer(8)::res06_8
!integer(8)::res07_8
!integer(8)::res08_8
!type(capture(k1=2_1,k2=4_2,k4=8_4,k8=1_8))::x2481
!integer(2)::res09_2
!integer(4)::res10_4
!integer(8)::res11_8
!integer(1)::res12_1
!type(capture(k1=2_1,k2=1_2,k4=8_4,k8=4_8))::x2184
!integer(2)::res13_2
!integer(1)::res14_1
!integer(8)::res15_8
!integer(4)::res16_4
!type(defaulted(n1=1_1,n2=2_2,n4=4_4,n8=8_8))::x1248
!integer(1)::res17_1
!integer(2)::res18_2
!integer(4)::res19_4
!integer(8)::res20_8
!type(defaulted(n1=2_1,n2=4_2,n4=8_4,n8=4_8))::x2484
!integer(2)::res21_2
!integer(4)::res22_4
!integer(8)::res23_8
!integer(4)::res24_4
!type(defaulted(n1=1_1,n2=2_2,n4=4_4,n8=2_8))::x1242
!integer(1)::res25_1
!integer(2)::res26_2
!integer(4)::res27_4
!integer(2)::res28_2
!type(extension(k5=1_4,n1=1_1,n2=1_2,n4=1_4,n8=1_8))::x11111
!integer(1)::res29_1
!integer(1)::res30_1
!integer(1)::res31_1
!integer(1)::res32_1
!integer(1)::res33_1
!type(extension(k5=8_4,n1=2_1,n2=8_2,n4=4_4,n8=1_8))::x28418
!integer(2)::res34_2
!integer(8)::res35_8
!integer(4)::res36_4
!integer(1)::res37_1
!integer(8)::res38_8
!type(extension(k5=2_4,n1=8_1,n2=4_2,n4=8_4,n8=1_8))::x84812
!integer(8)::res39_8
!integer(4)::res40_4
!integer(8)::res41_8
!integer(1)::res42_1
!integer(2)::res43_2
!type(extension(k5=2_4,n1=1_1,n2=2_2,n4=4_4,n8=8_8))::x12482
!integer(1)::res44_1
!integer(2)::res45_2
!integer(4)::res46_4
!integer(8)::res47_8
!integer(2)::res48_2
!end
