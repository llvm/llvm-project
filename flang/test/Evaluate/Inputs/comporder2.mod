!mod$ v1 sum:3235f4a02cdad423
!need$ 64657f78d85da21d n comporder1
module comporder2
use comporder1,only:base
type,extends(base)::intermediate
integer(4)::c2
end type
end
