module foo
  interface do_foo
    procedure do_foo_impl
  end interface
  interface do_bar
    procedure do_bar_impl
  end interface
contains
  subroutine do_foo_impl()
  end
  subroutine do_bar_impl()
  end
end
