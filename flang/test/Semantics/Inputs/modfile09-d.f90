submodule(m:s2) s3
  integer s3_x
end

!Expect: m-s3.mod
!submodule(m:s2) s3
!integer(4)::s3_x
!end
