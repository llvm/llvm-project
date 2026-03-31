module modfile84A
 contains
  subroutine foo()
  end subroutine
end
module modfile84B
 contains
  subroutine foo()
  end
end
module modfile84AB
  use modfile84A, only: foo, bar=>foo
  use modfile84B, only: foo, bar=>foo
end
