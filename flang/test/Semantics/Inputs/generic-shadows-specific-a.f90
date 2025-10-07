! these modules must be read from  module files
module m1
  interface f ! reference must be to generic
    module procedure f ! must have same name as generic interface
  end interface
 contains
  character function f() ! must be character
    f = 'q'
  end
end
module m2
   use m1
end
module m3
   use m2 ! must be m2, not m1
 contains
   subroutine mustExist() ! not called, but must exist
     character x
     x = f()
   end
end
