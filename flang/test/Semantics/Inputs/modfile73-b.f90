module modfile73ba
  use modfile73a, only: re_alloc, de_alloc
  charactermod_name
  type lData1D
     integer refCount  
     character   id  
     character  name  
  end type
  type TYPE_NAME
     type(lData1D), pointer :: data => null()
  end type
  interface refcount
  end interface
  interface initialized
  end interface
  interface same
  end interface
CONTAINS
   subroutine init_(this)
     end  
  subroutine delete_(this)
  end  
  subroutine assign_(this, other)
  end  
  function initialized_(thisinit)
  end  
  function same_(this1,this2same)
  end  
  function refcount_(thiscount)
  end  
  function id_(thisstr)
  end  
  function name_(this) result(str)
   type(TYPE_NAME)  this
   character(len_trim(this%data%name)) str
  end  
  subroutine tag_new_object(this)
  end  
  subroutine delete_Data(a1d_data)
  end 
end  

module modfile73bb
  use modfile73a, only: re_alloc, de_alloc
  charactermod_name
  type lData1D
     integer refCount  
     character   id  
     character  name  
logical, pointer       :: val => null()   
  end type
  TYPE_NAME
     type(lData1D), pointer :: data => null()
  end type
  PRIVATE
  public  TYPE_NAME
  public  initdelete, assignment, refcount, id
  public  name
  public  allocated
  interface init
  end interface
  interface delete
  end interface
  interface initialized
      subroutine die(str)
      end  
  end interface
CONTAINS
   subroutine init_(this)
  end  
  subroutine delete_(this)
  end  
  subroutine assign_(this, other)
  end  
  function initialized_(thisinit)
  end  
  function same_(this1,this2same)
  end  
  function refcount_(thiscount)
  end  
  function id_(thisstr)
  end  
  function name_(thisstr)
  end  
  subroutine tag_new_object(this)
  end  
  subroutine delete_Data(a1d_data)
  end 
end
