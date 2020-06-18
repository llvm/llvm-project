submodule(m:s1) s2
  integer s2_x
end

!Expect: m-s2.mod
!submodule(m:s1) s2
!integer(4)::s2_x
!end
