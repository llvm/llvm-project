! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

module shape_mod

  type shape
     integer :: color
     logical :: filled
     integer :: x
     integer :: y
   contains
     procedure,pass(this) :: write => write_shape
     procedure :: draw => draw_shape
  end type shape
  
  type, EXTENDS ( shape ) :: rectangle
  integer :: the_length
  integer :: the_width
contains
  procedure,pass(this) :: write => write_rec
  procedure :: draw => draw_rectangle
  procedure,pass(this) :: write_rec2
end type rectangle

interface
     subroutine draw_rectangle(this,results,i)
       import rectangle
       class (rectangle) :: this
       integer results(:)
       integer i
     end subroutine draw_rectangle
  end interface


type, extends (rectangle) :: square
contains
  !procedure :: draw => draw_sq
  !procedure,pass(this) :: write => write_sq
  procedure,pass(this) :: write_sq2
  
end type square
  interface
     subroutine write_rec2(results,i,this)
       import rectangle 
       class (rectangle) :: this
       integer i
       integer results(:)
     end subroutine write_rec2
  end interface
  interface
     subroutine write_sq(this,results,i)
       import square
       class (square) :: this
       integer results(:)
       integer i
     end subroutine write_sq
  end interface
  interface
     subroutine draw_sq(this,results,i)
       import square 
       class (square) :: this
       integer results(:)
       integer i
     end subroutine draw_sq
  end interface
  interface
     subroutine write_sq2(results,i,this)
       import square,rectangle
       class (square) :: this
       integer i
       integer results(:)
     end subroutine write_sq2
  end interface
  interface
     subroutine write_shape(this,results,i)
       import shape 
       class (shape) :: this
       integer results(:)
       integer i
     end subroutine write_shape
  end interface
  interface
     subroutine draw_shape(this,results,i)
       import  shape
       class (shape) :: this
       integer results(:)
       integer i
     end subroutine draw_shape
  end interface
  interface
     subroutine write_rec(this,results,i)
       import rectangle 
       class (rectangle) :: this
       integer results(:)
       integer i
     end subroutine write_rec
  end interface


end module shape_mod


subroutine write_shape(this,results,i)
  use :: shape_mod, except => write_shape
  class (shape) :: this
  integer results(:)
  integer i
  type(shape) :: sh
  results(i) = same_type_as(sh,this)
end subroutine write_shape

subroutine write_rec(this,results,i)
  use :: shape_mod, except => write_rec
  class (rectangle) :: this
  integer results(:)
  integer i
  type(rectangle) :: rec
  results(i) = same_type_as(rec,this)
end subroutine write_rec

subroutine draw_shape(this,results,i)
  use :: shape_mod, except => draw_shape
  class (rectangle) :: this
  integer results(:)
  integer i
end subroutine draw_shape

subroutine draw_rectangle(this,results,i)
  use :: shape_mod, except => draw_rectangle
  class (rectangle) :: this
  integer results(:)
  integer i
  type(rectangle) :: rec
  results(i) = same_type_as(this,rec)
end subroutine draw_rectangle

subroutine write_sq(this,results,i)
  use :: shape_mod, except => write_sq
  class (square) :: this
  integer results(:)
  integer i
  type(rectangle) :: rec
  results(i) = extends_type_of(this,rec)
end subroutine write_sq

subroutine draw_sq(this,results,i)
  use :: shape_mod, except => draw_sq
  class (square) :: this
  integer results(:)
  integer i
  type(rectangle) :: rec
  results(i) = extends_type_of(this,rec)
end subroutine draw_sq

subroutine write_sq2(results,i,this)
  use :: shape_mod, except => write_sq2
  class (rectangle) :: this
  integer i
  integer results(:)
  type(rectangle) :: rec
  type(square) :: sq
  results(i) = extends_type_of(this,rec)
end subroutine write_sq2

subroutine write_rec2(results,i,this)
  use :: shape_mod, except => write_rec2
  class (rectangle) :: this
  integer i
  integer results(:)
  type(rectangle) :: rec
  results(i) = same_type_as(this,rec)
end subroutine write_rec2




program p
USE CHECK_MOD
  use shape_mod
  
  logical l 
  integer results(6)
  integer expect(6)
  data expect /.true.,.false.,.false.,.true.,.true.,.true./
  data results /.false.,.true.,.true.,.false.,.false.,.false./
  class(square),allocatable :: s
  class(shape),allocatable :: sh
  type(rectangle) :: r
  
  allocate(s)
  s%the_length = 1000
  s%color = 1
  call s%write_sq2(results,1)
  call s%write(results,2)
  call s%draw(results,3)
  
  call s%rectangle%write(results,4);
  call draw_rectangle(s%rectangle,results,5)
  
  allocate(sh)
  call sh%write(results,6)
  
  call check(results,expect,6)
  
end program p


