!mod$ v1 sum:37cfecee3234c8ab
module modfile67
type::t
procedure(foo),nopass,pointer::p
end type
contains
pure function foo(n,a) result(r)
integer(4),intent(in)::n
real(4),intent(in)::a(1_8:int(n,kind=8))
logical(4)::r(1_8:int(int(max(0_8,int(n,kind=8)),kind=4),kind=8))
end
function fooptr(f)
procedure(foo)::f
type(t)::fooptr
end
end
