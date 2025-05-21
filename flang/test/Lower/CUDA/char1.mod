!mod$ v1 sum:dcc937f37583c496
module char1
character(1_8,1),allocatable::da(:)
attributes(device) da
character(1_8,1),allocatable::db(:)
attributes(device) db
contains
attributes(device) function check_char(c1,c2)
character(1_8,1),value::c1
character(1_8,1),value::c2
logical(4)::check_char
end
end
