!mod$ v1 sum:17d756d2fb56521c
module m
contains
subroutine func1(foo)
real(2)::foo
!dir$ ignore_tkr(d) foo
end
subroutine func3(foo)
real(2)::foo
!dir$ ignore_tkr(d) foo
end
subroutine func4(foo)
real(2)::foo
!dir$ ignore_tkr(d) foo
end
end
