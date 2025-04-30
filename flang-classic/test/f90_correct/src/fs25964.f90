! Test contributed by Neil Carlson from Los Alamos National Laboratory
! Related to github issue #240 

module map_type
type :: item
type(item), pointer :: next => null(), prev => null()
contains
final :: item_delete
end type
type :: map
type(item), pointer :: first => null()
end type
type :: parameter_list
type(map) :: params = map() ! flag rejects this valid empty constructor
end type
contains
subroutine item_delete(this)
type(item), intent(inout) :: this
end subroutine
end module
