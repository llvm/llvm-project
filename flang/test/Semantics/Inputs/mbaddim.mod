!mod$ v1 sum:0df54fe14f12f633
module mbaddim
contains
subroutine s(n,m,c,a,b,d)
integer(4),intent(in)::n(1_8:2_8)
integer(4),intent(in)::m(1_8:2_8)
logical(4),intent(in)::c(1_8:2_8)
real(4)::a(1_8:__builtin_int(merge(n,m,c),kind=8))
real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,ubound(a,dim=9,kind=8)),kind=4),kind=8))
real(4)::d(1_8:__builtin_int(__builtin_int(max(0_8,ubound(a,dim=2147483648_8,kind=8)),kind=4),kind=8))
end
end
