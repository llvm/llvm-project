!mod$ v1 sum:dc22d009f1dc3500
module globals
real(4),allocatable,device::a_device(:)
real(4),allocatable,managed::a_managed(:)
real(4),allocatable,pinned::a_pinned(:)
type::t1
integer(4)::a
real(4),allocatable,device::b(:)
end type
end
